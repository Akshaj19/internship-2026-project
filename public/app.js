/**
 * Akshaj Motion Gesture AI
 * Production dashboard — Edge Impulse + DeviceMotion integration
 */

(function () {
  'use strict';

  /* ==========================================================================
     Configuration
     ========================================================================== */
  const CONFIG = {
    chartHistoryLength: 120,
    maxConsoleLines: 200,
    gestureDebounceMs: 800,
    accelBarRange: 20,
    defaultFrequency: 62.5,
    defaultInputFeatures: 375,
    displayNames: {
      idle: 'IDLE',
      clockwise: 'CLOCKWISE',
      anticlockwise: 'ANTI-CLOCKWISE',
      wave: 'WAVE'
    }
  };

  /* ==========================================================================
     Application State
     ========================================================================== */
  const state = {
    sensorsActive: false,
    detecting: false,
    classifierReady: false,
    permissionGranted: false,
    currentGesture: 'idle',
    currentDisplayLabel: 'IDLE',
    currentConfidence: 0,
    axis: { x: 0, y: 0, z: 0 },
    sampleBuffer: [],
    inputFeaturesCount: CONFIG.defaultInputFeatures,
    sliceSize: 0,
    frequency: CONFIG.defaultFrequency,
    intervalMs: 16,
    useContinuousMode: false,
    classificationThreshold: 0.6,
    modelLabels: [],
    currentSample: { x: 0, y: 0, z: 0 },
    sampleTimer: null,
    updateRate: 0,
    lastSampleTime: 0,
    rateSamples: [],
    isClassifying: false,
    csvRows: [],
    analytics: {
      total: 0,
      clockwise: 0,
      anticlockwise: 0,
      wave: 0,
      confidenceSum: 0,
      confidenceCount: 0,
      sessionStart: Date.now(),
      lastGestureTime: 0
    },
    lastClassifyTime: 0,
    chartData: { x: [], y: [], z: [] }
  };

  /* ==========================================================================
     DOM References
     ========================================================================== */
  const DOM = {};

  function cacheDOM() {
    const ids = [
      'navLiveDot', 'navLiveText', 'liveStatusText', 'sensorStatusText', 'modelStatusText',
      'liveStatusPill', 'sensorStatusPill', 'modelStatusPill',
      'btnActivateSensors', 'btnStartDetection', 'btnDownloadCsv', 'btnClearConsole',
      'gestureLabel', 'confidenceValue', 'confidenceRing',
      'axisX', 'axisY', 'axisZ', 'barX', 'barY', 'barZ',
      'chartFps', 'accelChart', 'debugConsole',
      'statTotal', 'statClockwise', 'statAnticlockwise', 'statWave',
      'statAvgConfidence', 'statDetectionRate',
      'healthScore', 'healthSummary',
      'healthPermission', 'healthBrowser', 'healthHttps', 'healthAvailability', 'healthRate'
    ];

    ids.forEach(function (id) {
      DOM[id] = document.getElementById(id);
    });

    DOM.predictionCard = document.querySelector('.prediction-card');
    DOM.legendItems = document.querySelectorAll('.legend-item');
  }

  /* ==========================================================================
     Logger — Debug Console
     ========================================================================== */
  const Logger = {
    log: function (message, type) {
      type = type || 'info';
      var consoleEl = DOM.debugConsole;
      if (!consoleEl) return;

      var time = new Date().toLocaleTimeString('en-US', { hour12: false });
      var line = document.createElement('div');
      line.className = 'log-line log-' + type;
      line.innerHTML =
        '<span class="log-time">[' + time + ']</span>' +
        '<span class="log-msg">' + escapeHtml(String(message)) + '</span>';

      consoleEl.appendChild(line);

      while (consoleEl.children.length > CONFIG.maxConsoleLines) {
        consoleEl.removeChild(consoleEl.firstChild);
      }

      consoleEl.scrollTop = consoleEl.scrollHeight;
    },

    clear: function () {
      if (DOM.debugConsole) {
        DOM.debugConsole.innerHTML = '';
        this.log('Console cleared', 'info');
      }
    }
  };

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /* ==========================================================================
     Edge Impulse Classifier Wrapper
     ========================================================================== */
  const ClassifierManager = {
    classifier: null,
    initialized: false,

    init: function () {
      var self = this;
      Logger.log('Initializing Edge Impulse WASM classifier…', 'info');
      setStatusIndicator('modelStatusPill', 'loading');
      setText('modelStatusText', 'Loading Model');

      return new Promise(function (resolve) {
        var resolved = false;

        function onReady() {
          if (resolved) return;
          resolved = true;

          try {
            self.classifier = new EdgeImpulseClassifier();
            self.classifier.init().then(function () {
              self.initialized = true;
              state.classifierReady = true;

              var props = self.classifier.getProperties();
              if (props) {
                state.inputFeaturesCount = props.inputFeaturesCount;
                state.frequency = props.frequency;
                state.intervalMs = props.intervalMs || Math.round(1000 / props.frequency);
                state.useContinuousMode = props.useContinuousMode;
                state.sliceSize = props.sliceSize;
                state.classificationThreshold = props.classificationThreshold;

                Logger.log('Model: ' + props.inputFeaturesCount + ' features @ ' + props.frequency + ' Hz', 'success');
                Logger.log('Mode: ' + (props.useContinuousMode ? 'continuous (slice ' + props.sliceSize + ')' : 'window'), 'info');
                Logger.log('Threshold: ' + Math.round(props.classificationThreshold * 100) + '%', 'info');
              } else {
                Logger.log('Model loaded with default settings', 'success');
              }

              setStatusIndicator('modelStatusPill', 'active');
              setText('modelStatusText', 'Ready');
              DOM.btnStartDetection.disabled = !state.sensorsActive;
              resolve(true);
            }).catch(function (err) {
              self.handleInitError(err);
              resolve(false);
            });
          } catch (err) {
            self.handleInitError(err);
            resolve(false);
          }
        }

        if (typeof Module !== 'undefined') {
          if (Module.calledRun) {
            onReady();
          } else {
            var prev = Module.onRuntimeInitialized;
            Module.onRuntimeInitialized = function () {
              if (typeof prev === 'function') prev();
              onReady();
            };
            setTimeout(function () {
              if (!resolved && typeof Module.run_classifier === 'function') {
                onReady();
              }
            }, 8000);
          }
        } else {
          self.handleInitError(new Error('Edge Impulse Module not found'));
          resolve(false);
        }
      });
    },

    handleInitError: function (err) {
      state.classifierReady = false;
      setStatusIndicator('modelStatusPill', 'error');
      setText('modelStatusText', 'Load Failed');
      Logger.log('Classifier init failed: ' + (err.message || err), 'error');
      Logger.log('Ensure edge-impulse-standalone.wasm is deployed alongside the JS file', 'warn');
    },

    classify: function (buffer) {
      if (!this.initialized || !this.classifier || state.isClassifying) return null;
      state.isClassifying = true;
      try {
        var data = buffer.slice(buffer.length - state.inputFeaturesCount);
        var result;

        if (state.useContinuousMode && state.sliceSize > 0) {
          var slice = data.slice(data.length - state.sliceSize);
          result = this.classifier.classifyContinuous(slice, false, true);
        } else {
          result = this.classifier.classify(data, false);
        }

        return result;
      } catch (err) {
        Logger.log('Classification error: ' + err.message, 'error');
        return null;
      } finally {
        state.isClassifying = false;
      }
    }
  };

  function EdgeImpulseClassifier() {
    this._initialized = false;
    this._props = null;
  }

  var classifierInitialized = false;

  EdgeImpulseClassifier.prototype.init = function () {
    if (classifierInitialized) return Promise.resolve();

    return new Promise(function (resolve, reject) {
      if (typeof Module === 'undefined') {
        reject(new Error('Module is undefined'));
        return;
      }

      function finishInit() {
        if (typeof Module.init === 'function') {
          var initCode = Module.init();
          if (typeof initCode === 'number' && initCode !== 0) {
            reject(new Error('Module.init() failed with code ' + initCode));
            return;
          }
        }

        if (typeof Module.run_classifier !== 'function') {
          reject(new Error('run_classifier not exported — check WASM deployment'));
          return;
        }

        classifierInitialized = true;
        resolve();
      }

      if (Module.calledRun) {
        finishInit();
      } else {
        var prev = Module.onRuntimeInitialized;
        Module.onRuntimeInitialized = function () {
          if (typeof prev === 'function') prev();
          finishInit();
        };
      }
    });
  };

  EdgeImpulseClassifier.prototype.classify = function (rawData, debug) {
    if (!classifierInitialized) throw new Error('Classifier not initialized');

    var heap = this._arrayToHeap(rawData);
    var ret = Module.run_classifier(heap.buffer.byteOffset, rawData.length, debug || false);
    Module._free(heap.ptr);

    if (ret.result !== 0) {
      throw new Error('Classification failed (code: ' + ret.result + ')');
    }

    return this._parseResult(ret);
  };

  EdgeImpulseClassifier.prototype.classifyContinuous = function (rawData, debug, enablePerfCal) {
    if (!classifierInitialized) throw new Error('Classifier not initialized');

    if (typeof Module.run_classifier_continuous !== 'function') {
      return this.classify(rawData, debug);
    }

    var heap = this._arrayToHeap(rawData);
    var ret = Module.run_classifier_continuous(
      heap.buffer.byteOffset,
      rawData.length,
      debug || false,
      enablePerfCal !== false
    );
    Module._free(heap.ptr);

    if (ret.result !== 0) {
      throw new Error('Continuous classification failed (code: ' + ret.result + ')');
    }

    return this._parseResult(ret);
  };

  EdgeImpulseClassifier.prototype._parseResult = function (ret) {
    var result = { anomaly: ret.anomaly, results: [] };

    for (var i = 0; i < ret.size(); i++) {
      var c = ret.get(i);
      result.results.push({ label: c.label, value: c.value });
      c.delete();
    }

    ret.delete();
    return result;
  };

  EdgeImpulseClassifier.prototype.getProperties = function () {
    if (typeof Module.get_properties !== 'function') return null;

    var ret = Module.get_properties();
    return {
      inputFeaturesCount: ret.input_features_count || CONFIG.defaultInputFeatures,
      frequency: ret.frequency || CONFIG.defaultFrequency,
      intervalMs: ret.interval_ms || Math.round(1000 / (ret.frequency || CONFIG.defaultFrequency)),
      axisCount: ret.axis_count || 3,
      useContinuousMode: !!ret.use_continuous_mode,
      sliceSize: ret.slice_size || 0,
      classificationThreshold: typeof ret.classification_threshold === 'number'
        ? ret.classification_threshold
        : 0.6
    };
  };

  EdgeImpulseClassifier.prototype._arrayToHeap = function (data) {
    var typed = new Float32Array(data);
    var numBytes = typed.length * typed.BYTES_PER_ELEMENT;
    var ptr = Module._malloc(numBytes);
    var heapBytes = new Uint8Array(Module.HEAPU8.buffer, ptr, numBytes);
    heapBytes.set(new Uint8Array(typed.buffer));
    return { ptr: ptr, buffer: heapBytes };
  };

  /* ==========================================================================
     Gesture Label Normalization
     ========================================================================== */
  function normalizeGesture(label) {
    if (!label) return 'idle';
    var lower = label.toLowerCase().replace(/[\s-]+/g, '_');

    if (lower.indexOf('ccw') !== -1 || (lower.indexOf('anti') !== -1 && lower.indexOf('clock') !== -1)) {
      return 'anticlockwise';
    }
    if (lower.indexOf('clock') !== -1 || lower.indexOf('_cw') !== -1 || lower === 'cw') {
      return 'clockwise';
    }
    if (lower.indexOf('wave') !== -1 || lower.indexOf('waving') !== -1) return 'wave';
    if (lower.indexOf('idle') !== -1 || lower.indexOf('rest') !== -1 || lower.indexOf('stationary') !== -1) {
      return 'idle';
    }

    return lower;
  }

  function formatDisplayLabel(label) {
    if (!label) return 'IDLE';
    var key = normalizeGesture(label);
    if (CONFIG.displayNames[key]) return CONFIG.displayNames[key];
    return label.replace(/[_-]/g, ' ').toUpperCase();
  }

  function getTopPrediction(results) {
    if (!results || !results.length) return { gesture: 'idle', rawLabel: 'idle', confidence: 0, allResults: [] };

    var sorted = results.slice().sort(function (a, b) { return b.value - a.value; });

    if (state.modelLabels.length === 0) {
      state.modelLabels = results.map(function (r) { return r.label; });
      Logger.log('Model labels: ' + state.modelLabels.join(', '), 'success');
    }

    var hasSignal = sorted.some(function (r) { return r.value > 0.01; });
    if (!hasSignal) {
      return { gesture: 'idle', rawLabel: 'idle', confidence: 0, allResults: sorted };
    }

    var best = sorted[0];
    var aboveThreshold = sorted.find(function (r) {
      return r.value >= state.classificationThreshold;
    });

    var pick = aboveThreshold || best;

    return {
      gesture: normalizeGesture(pick.label),
      rawLabel: pick.label,
      confidence: Math.round(pick.value * 100),
      allResults: sorted
    };
  }

  /* ==========================================================================
     Sensor Manager — DeviceMotion API
     ========================================================================== */
  const SensorManager = {
    motionHandler: null,

    isSupported: function () {
      return typeof window !== 'undefined' &&
        ('DeviceMotionEvent' in window || 'ondevicemotion' in window);
    },

    isSecureContext: function () {
      return window.isSecureContext ||
        location.protocol === 'https:' ||
        location.hostname === 'localhost' ||
        location.hostname === '127.0.0.1';
    },

    requestPermission: function () {
      var self = this;

      if (typeof DeviceMotionEvent !== 'undefined' &&
          typeof DeviceMotionEvent.requestPermission === 'function') {
        return DeviceMotionEvent.requestPermission()
          .then(function (response) {
            state.permissionGranted = response === 'granted';
            return state.permissionGranted;
          })
          .catch(function () {
            state.permissionGranted = false;
            return false;
          });
      }

      state.permissionGranted = true;
      return Promise.resolve(true);
    },

    activate: function () {
      var self = this;

      if (!this.isSupported()) {
        Logger.log('DeviceMotion API not supported in this browser', 'error');
        return Promise.resolve(false);
      }

      if (!this.isSecureContext()) {
        Logger.log('Accelerometer requires HTTPS or localhost', 'warn');
      }

      return this.requestPermission().then(function (granted) {
        if (!granted) {
          Logger.log('Motion sensor permission denied', 'error');
          updateHealth();
          return false;
        }

        Logger.log('Motion sensor permission granted', 'success');
        self.startListening();
        state.sensorsActive = true;
        setStatusIndicator('sensorStatusPill', 'active');
        setText('sensorStatusText', 'Active');
        DOM.btnStartDetection.disabled = !state.classifierReady;
        updateHealth();
        return true;
      });
    },

    startListening: function () {
      var self = this;

      if (this.motionHandler) return;

      this.motionHandler = function (event) {
        self.onMotion(event);
      };

      window.addEventListener('devicemotion', this.motionHandler, { passive: true });
      this.startSampling();
      Logger.log('Accelerometer stream started @ ' + state.frequency + ' Hz', 'sensor');
    },

    stopListening: function () {
      if (this.motionHandler) {
        window.removeEventListener('devicemotion', this.motionHandler);
        this.motionHandler = null;
      }
      this.stopSampling();
      state.sensorsActive = false;
      setStatusIndicator('sensorStatusPill', 'idle');
      setText('sensorStatusText', 'Inactive');
    },

    startSampling: function () {
      var self = this;
      this.stopSampling();

      var intervalMs = Math.round(1000 / state.frequency);
      state.sampleTimer = setInterval(function () {
        self.captureSample();
      }, intervalMs);
    },

    stopSampling: function () {
      if (state.sampleTimer) {
        clearInterval(state.sampleTimer);
        state.sampleTimer = null;
      }
    },

    onMotion: function (event) {
      var acc = event.accelerationIncludingGravity;
      if (!acc || (acc.x === null && acc.y === null && acc.z === null)) {
        acc = event.acceleration;
      }
      if (!acc) return;

      state.currentSample = {
        x: acc.x || 0,
        y: acc.y || 0,
        z: acc.z || 0
      };
    },

    captureSample: function () {
      var x = state.currentSample.x;
      var y = state.currentSample.y;
      var z = state.currentSample.z;

      state.axis = { x: x, y: y, z: z };

      var now = performance.now();
      if (state.lastSampleTime) {
        state.rateSamples.push(1000 / (now - state.lastSampleTime));
        if (state.rateSamples.length > 30) state.rateSamples.shift();
        state.updateRate = Math.round(
          state.rateSamples.reduce(function (a, b) { return a + b; }, 0) / state.rateSamples.length
        );
      }
      state.lastSampleTime = now;

      state.sampleBuffer.push(x, y, z);
      while (state.sampleBuffer.length > state.inputFeaturesCount) {
        state.sampleBuffer.shift();
        state.sampleBuffer.shift();
        state.sampleBuffer.shift();
      }

      UI.updateSensorDisplay(x, y, z);
      ChartEngine.push(x, y, z);

      if (state.detecting) {
        var timestamp = new Date().toISOString();
        state.csvRows.push({
          timestamp: timestamp,
          x: x.toFixed(4),
          y: y.toFixed(4),
          z: z.toFixed(4),
          gesture: state.currentDisplayLabel || 'IDLE',
          confidence: state.currentConfidence
        });

        if (state.sampleBuffer.length >= state.inputFeaturesCount) {
          DetectionEngine.runClassification();
        }
      }

      updateHealth();
    }
  };

  /* ==========================================================================
     Detection Engine
     ========================================================================== */
  const DetectionEngine = {
    start: function () {
      if (!state.sensorsActive) {
        Logger.log('Activate sensors before starting detection', 'warn');
        return;
      }

      if (!state.classifierReady) {
        Logger.log('Classifier not ready — check Edge Impulse deployment', 'warn');
        return;
      }

      state.detecting = true;
      state.analytics.sessionStart = Date.now();
      DOM.btnStartDetection.textContent = '';
      DOM.btnStartDetection.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Stop Detection';

      setStatusIndicator('liveStatusPill', 'active');
      setText('liveStatusText', 'Detecting');
      setNavLive(true);
      DOM.predictionCard.classList.add('detecting');

      Logger.log('Gesture detection started', 'success');
      Logger.log('Perform gestures continuously for 2+ seconds per motion', 'info');
    },

    stop: function () {
      state.detecting = false;

      DOM.btnStartDetection.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Detection';

      setStatusIndicator('liveStatusPill', 'idle');
      setText('liveStatusText', 'Standby');
      setNavLive(false);
      DOM.predictionCard.classList.remove('detecting');

      Logger.log('Gesture detection stopped', 'info');
    },

    toggle: function () {
      if (state.detecting) this.stop();
      else this.start();
    },

    runClassification: function () {
      if (!state.detecting || !state.classifierReady) return;

      var now = Date.now();
      if (now - state.lastClassifyTime < 200) return;
      state.lastClassifyTime = now;

      if (state.sampleBuffer.length < state.inputFeaturesCount) {
        var pct = Math.round((state.sampleBuffer.length / state.inputFeaturesCount) * 100);
        setText('liveStatusText', 'Buffering ' + pct + '%');
        return;
      }

      var result = ClassifierManager.classify(state.sampleBuffer);
      if (!result || !result.results.length) return;

      var prediction = getTopPrediction(result.results);

      UI.updatePrediction(prediction.gesture, prediction.rawLabel, prediction.confidence);

      var allScores = prediction.allResults.map(function (r) {
        return r.label + ':' + Math.round(r.value * 100) + '%';
      }).join(' | ');

      Logger.log(allScores, 'prediction');

      if (prediction.gesture !== 'idle' &&
          prediction.confidence >= Math.round(state.classificationThreshold * 100) &&
          now - state.analytics.lastGestureTime > CONFIG.gestureDebounceMs) {

        Analytics.recordGesture(prediction.gesture, prediction.confidence);
        state.analytics.lastGestureTime = now;
      }
    }
  };

  /* ==========================================================================
     Analytics
     ========================================================================== */
  const Analytics = {
    recordGesture: function (gesture, confidence) {
      if (gesture === 'idle') return;

      state.analytics.total++;
      if (gesture === 'clockwise') state.analytics.clockwise++;
      else if (gesture === 'anticlockwise') state.analytics.anticlockwise++;
      else if (gesture === 'wave') state.analytics.wave++;

      state.analytics.confidenceSum += confidence;
      state.analytics.confidenceCount++;

      UI.updateAnalytics();
      UI.flashStat(gesture);
    },

    getAvgConfidence: function () {
      if (!state.analytics.confidenceCount) return 0;
      return Math.round(state.analytics.confidenceSum / state.analytics.confidenceCount);
    },

    getDetectionRate: function () {
      var elapsedMin = (Date.now() - state.analytics.sessionStart) / 60000;
      if (elapsedMin < 0.01) return 0;
      return Math.round(state.analytics.total / elapsedMin);
    }
  };

  /* ==========================================================================
     Chart Engine — Canvas Line Chart
     ========================================================================== */
  const ChartEngine = {
    canvas: null,
    ctx: null,
    animFrame: null,

    init: function () {
      this.canvas = DOM.accelChart;
      if (!this.canvas) return;
      this.ctx = this.canvas.getContext('2d');
      this.resize();
      window.addEventListener('resize', debounce(this.resize.bind(this), 150));
      this.loop();
    },

    resize: function () {
      if (!this.canvas) return;
      var wrap = this.canvas.parentElement;
      var dpr = window.devicePixelRatio || 1;
      var w = wrap.clientWidth;
      var h = wrap.clientHeight;
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
      this.canvas.style.width = w + 'px';
      this.canvas.style.height = h + 'px';
      if (this.ctx) this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    },

    push: function (x, y, z) {
      var len = CONFIG.chartHistoryLength;
      state.chartData.x.push(x);
      state.chartData.y.push(y);
      state.chartData.z.push(z);

      if (state.chartData.x.length > len) {
        state.chartData.x.shift();
        state.chartData.y.shift();
        state.chartData.z.shift();
      }
    },

    loop: function () {
      var self = this;
      self.draw();
      self.animFrame = requestAnimationFrame(function () { self.loop(); });
    },

    draw: function () {
      if (!this.ctx || !this.canvas) return;

      var ctx = this.ctx;
      var w = this.canvas.clientWidth;
      var h = this.canvas.clientHeight;
      var pad = { top: 16, right: 16, bottom: 24, left: 40 };
      var plotW = w - pad.left - pad.right;
      var plotH = h - pad.top - pad.bottom;

      ctx.clearRect(0, 0, w, h);

      ctx.fillStyle = 'rgba(255,255,255,0.03)';
      for (var g = 0; g <= 4; g++) {
        var gy = pad.top + (plotH / 4) * g;
        ctx.fillRect(pad.left, gy, plotW, 1);
      }

      var data = state.chartData;
      if (data.x.length < 2) return;

      var all = data.x.concat(data.y, data.z);
      var min = Math.min.apply(null, all);
      var max = Math.max.apply(null, all);
      var range = max - min || 1;
      min -= range * 0.1;
      max += range * 0.1;
      range = max - min;

      var colors = {
        x: '#00E5FF',
        y: '#7C4DFF',
        z: '#00FFA3'
      };

      ['x', 'y', 'z'].forEach(function (axis) {
        self.drawLine(ctx, data[axis], colors[axis], pad, plotW, plotH, min, range);
      });

      if (DOM.chartFps && state.updateRate) {
        DOM.chartFps.textContent = state.updateRate + ' Hz';
      }
    },

    drawLine: function (ctx, points, color, pad, plotW, plotH, min, range) {
      var len = points.length;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';

      for (var i = 0; i < len; i++) {
        var x = pad.left + (i / (len - 1)) * plotW;
        var y = pad.top + plotH - ((points[i] - min) / range) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      ctx.stroke();

      ctx.lineTo(pad.left + plotW, pad.top + plotH);
      ctx.lineTo(pad.left, pad.top + plotH);
      ctx.closePath();
      var grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
      grad.addColorStop(0, color.replace(')', ', 0.15)').replace('rgb', 'rgba').replace('#', ''));
      ctx.fillStyle = this.hexToRgba(color, 0.08);
      ctx.fill();
    },

    hexToRgba: function (hex, alpha) {
      var r = parseInt(hex.slice(1, 3), 16);
      var g = parseInt(hex.slice(3, 5), 16);
      var b = parseInt(hex.slice(5, 7), 16);
      return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
    }
  };

  /* ==========================================================================
     CSV Export
     ========================================================================== */
  const CSVExporter = {
    download: function () {
      if (!state.csvRows.length) {
        Logger.log('No sensor data to export — start detection first', 'warn');
        return;
      }

      var header = 'timestamp,x,y,z,gesture,confidence\n';
      var rows = state.csvRows.map(function (r) {
        return [r.timestamp, r.x, r.y, r.z, r.gesture, r.confidence].join(',');
      }).join('\n');

      var blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'akshaj-gesture-data-' + Date.now() + '.csv';
      a.click();
      URL.revokeObjectURL(url);

      Logger.log('Exported ' + state.csvRows.length + ' rows to CSV', 'success');
    }
  };

  /* ==========================================================================
     UI Updates
     ========================================================================== */
  const UI = {
    updateSensorDisplay: function (x, y, z) {
      setText('axisX', x.toFixed(2));
      setText('axisY', y.toFixed(2));
      setText('axisZ', z.toFixed(2));

      var range = CONFIG.accelBarRange;
      setBar('barX', ((x + range) / (range * 2)) * 100);
      setBar('barY', ((y + range) / (range * 2)) * 100);
      setBar('barZ', ((z + range) / (range * 2)) * 100);
    },

    updatePrediction: function (gesture, rawLabel, confidence) {
      state.currentGesture = gesture;
      state.currentDisplayLabel = formatDisplayLabel(rawLabel || gesture);
      state.currentConfidence = confidence;

      var labelEl = DOM.gestureLabel;
      if (labelEl) {
        labelEl.textContent = state.currentDisplayLabel;
        var cssKey = ['idle', 'clockwise', 'anticlockwise', 'wave'].indexOf(gesture) !== -1
          ? gesture
          : 'idle';
        labelEl.className = 'prediction-label gesture-' + cssKey;
      }

      setText('confidenceValue', confidence);

      var ring = DOM.confidenceRing;
      if (ring) {
        var offset = 879.65 - (confidence / 100) * 879.65;
        ring.style.strokeDashoffset = offset;
        var glow = document.querySelector('.ring-glow');
        if (glow) glow.style.strokeDashoffset = offset;
      }

      DOM.legendItems.forEach(function (item) {
        item.classList.toggle('active', item.dataset.gesture === gesture);
      });

      if (state.detecting) {
        setText('liveStatusText', 'Detecting');
      }
    },

    updateAnalytics: function () {
      setText('statTotal', state.analytics.total);
      setText('statClockwise', state.analytics.clockwise);
      setText('statAnticlockwise', state.analytics.anticlockwise);
      setText('statWave', state.analytics.wave);
      setText('statAvgConfidence', Analytics.getAvgConfidence() + '%');
      setText('statDetectionRate', Analytics.getDetectionRate() + '/min');
    },

    flashStat: function (gesture) {
      var map = {
        clockwise: 'statClockwise',
        anticlockwise: 'statAnticlockwise',
        wave: 'statWave'
      };
      var el = DOM[map[gesture]] || DOM.statTotal;
      if (el) {
        el.classList.add('updated');
        setTimeout(function () { el.classList.remove('updated'); }, 400);
      }
    }
  };

  /* ==========================================================================
     Sensor Health Panel
     ========================================================================== */
  function updateHealth() {
    var checks = [];
    var score = 0;
    var total = 5;

    var permOk = state.permissionGranted;
    setHealthItem('healthPermission', permOk ? 'Granted' : (state.sensorsActive ? 'Granted' : 'Not Granted'), permOk);
    if (permOk) score++;

    var browserOk = SensorManager.isSupported();
    setHealthItem('healthBrowser', browserOk ? 'Supported' : 'Unsupported', browserOk);
    if (browserOk) score++;

    var httpsOk = SensorManager.isSecureContext();
    setHealthItem('healthHttps', httpsOk ? 'Secure' : 'Insecure (HTTP)', httpsOk);
    if (httpsOk) score++;

    var availOk = state.sensorsActive && state.updateRate > 0;
    setHealthItem('healthAvailability', availOk ? 'Available' : (state.sensorsActive ? 'Waiting…' : 'Inactive'), availOk || state.sensorsActive);
    if (availOk) score++;

    var rateText = state.updateRate ? state.updateRate + ' Hz' : '0 Hz';
    var rateOk = state.updateRate >= 10;
    setHealthItem('healthRate', rateText, rateOk);
    if (rateOk) score++;

    var pct = Math.round((score / total) * 100);
    var scoreEl = DOM.healthScore;
    if (scoreEl) {
      scoreEl.textContent = pct + '%';
      scoreEl.className = 'health-score ' + (pct >= 80 ? 'good' : pct >= 50 ? 'warn' : 'bad');
    }

    var summary = DOM.healthSummary;
    if (summary) {
      if (pct >= 80) summary.textContent = 'All systems operational';
      else if (pct >= 50) summary.textContent = 'Some checks need attention';
      else summary.textContent = 'Environment not ready for detection';
    }
  }

  function setHealthItem(id, value, pass) {
    var el = DOM[id];
    if (!el) return;
    el.querySelector('.health-item-value').textContent = value;
    el.classList.remove('pass', 'fail', 'pending');
    if (pass === true) {
      el.classList.add('pass');
      el.querySelector('.health-check-icon').textContent = '✓';
    } else if (pass === false) {
      el.classList.add('fail');
      el.querySelector('.health-check-icon').textContent = '✗';
    } else {
      el.classList.add('pending');
      el.querySelector('.health-check-icon').textContent = '○';
    }
  }

  /* ==========================================================================
     Helpers
     ========================================================================== */
  function setText(id, text) {
    var el = DOM[id];
    if (el) el.textContent = text;
  }

  function setBar(id, percent) {
    var el = DOM[id];
    if (el) el.style.width = Math.max(0, Math.min(100, percent)) + '%';
  }

  function setStatusIndicator(pillId, status) {
    var pill = DOM[pillId];
    if (!pill) return;
    var indicator = pill.querySelector('.status-indicator');
    if (indicator) indicator.setAttribute('data-status', status);
  }

  function setNavLive(active) {
    if (DOM.navLiveDot) DOM.navLiveDot.classList.toggle('active', active);
    if (DOM.navLiveText) DOM.navLiveText.textContent = active ? 'Live' : 'Standby';
  }

  function debounce(fn, ms) {
    var timer;
    return function () {
      var args = arguments;
      var ctx = this;
      clearTimeout(timer);
      timer = setTimeout(function () { fn.apply(ctx, args); }, ms);
    };
  }

  /* ==========================================================================
     Event Bindings
     ========================================================================== */
  function bindEvents() {
    DOM.btnActivateSensors.addEventListener('click', function () {
      Logger.log('Requesting sensor activation…', 'info');
      SensorManager.activate().then(function (ok) {
        if (ok) Logger.log('Sensors activated — ready for detection', 'success');
      });
    });

    DOM.btnStartDetection.addEventListener('click', function () {
      DetectionEngine.toggle();
    });

    DOM.btnDownloadCsv.addEventListener('click', function () {
      CSVExporter.download();
    });

    DOM.btnClearConsole.addEventListener('click', function () {
      Logger.clear();
    });

    DOM.legendItems.forEach(function (item) {
      item.addEventListener('click', function () {
        var gesture = item.dataset.gesture;
        UI.updatePrediction(gesture, gesture, state.currentConfidence);
        Logger.log('Legend highlight: ' + formatDisplayLabel(gesture), 'info');
      });
    });
  }

  /* ==========================================================================
     App Bootstrap
     ========================================================================== */
  const App = {
    init: function () {
      cacheDOM();
      bindEvents();
      ChartEngine.init();
      updateHealth();

      UI.updatePrediction('idle', 'idle', 0);
      UI.updateAnalytics();
      UI.updateSensorDisplay(0, 0, 0);

      Logger.log('Akshaj Motion Gesture AI dashboard initialized', 'success');
      Logger.log('Tap "Activate Sensors" on your mobile device to begin', 'info');

      ClassifierManager.init().then(function (ready) {
        if (ready) {
          DOM.btnStartDetection.disabled = !state.sensorsActive;
        }
        updateHealth();
      });

      setInterval(function () {
        if (state.detecting) UI.updateAnalytics();
      }, 5000);
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', App.init);
  } else {
    App.init();
  }

})();
