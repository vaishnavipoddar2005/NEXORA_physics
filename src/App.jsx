import React, { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Environment, Line, OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import { Files, Camera, CameraOff, Hand, RotateCcw, Play, Pause, Ruler, Activity, GraduationCap, Info, CheckCircle2, AlertCircle, Move3D } from 'lucide-react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import * as THREE from 'three';

const WASM_URL = '/mediapipe/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const GRAVITY = 9.81;

function clamp(n, min, max) { return Math.max(min, Math.min(max, n)); }
function distance2D(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function useHandTracking(videoRef, onHands) {
  const landmarkerRef = useRef(null);
  const rafRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const onHandsRef = useRef(onHands);
  const [cameraStatus, setCameraStatus] = useState('idle');
  const [cameraError, setCameraError] = useState('');
  const streamRef = useRef(null);

  useEffect(() => { onHandsRef.current = onHands; }, [onHands]);

  const stopCamera = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    onHandsRef.current([]);
    setCameraStatus('idle');
  }, [videoRef]);

  const startCamera = useCallback(async () => {
    setCameraError('');
    setCameraStatus('loading');
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera access is unavailable. Open this app on localhost or HTTPS in a modern browser.');
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      if (!landmarkerRef.current) {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        }).catch(async () => {
          const vision = await FilesetResolver.forVisionTasks(WASM_URL);
          return HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
            runningMode: 'VIDEO',
            numHands: 2,
            minHandDetectionConfidence: 0.55,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
          });
        });
      }

      setCameraStatus('active');
      const detect = () => {
        if (!videoRef.current || !landmarkerRef.current || videoRef.current.readyState < 2) {
          rafRef.current = requestAnimationFrame(detect);
          return;
        }
        if (videoRef.current.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = videoRef.current.currentTime;
          const result = landmarkerRef.current.detectForVideo(videoRef.current, performance.now());
          const hands = (result.landmarks || []).map((landmarks, index) => {
            const handedness = result.handednesses?.[index]?.[0]?.categoryName || `Hand ${index + 1}`;
            const thumb = landmarks[4];
            const indexTip = landmarks[8];
            const pinchDistance = Math.hypot(thumb.x - indexTip.x, thumb.y - indexTip.y);
            return {
              id: handedness,
              landmarks,
              // Mirror x so gestures match the mirrored preview and feel natural.
              x: 1 - indexTip.x,
              y: indexTip.y,
              pinchDistance,
              pinch: pinchDistance < 0.065
            };
          });
          onHandsRef.current(hands);
        }
        rafRef.current = requestAnimationFrame(detect);
      };
      rafRef.current = requestAnimationFrame(detect);
    } catch (error) {
      console.error(error);
      setCameraError(error?.message || 'Could not start the camera. Check browser permission and try again.');
      setCameraStatus('error');
      if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  }, [videoRef]);

  useEffect(() => () => {
    cancelAnimationFrame(rafRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
    landmarkerRef.current?.close();
  }, []);

  return { cameraStatus, cameraError, startCamera, stopCamera };
}

function PendulumModel({ angle, length, dragging, bobPositionRef }) {
  const pivot = useMemo(() => new THREE.Vector3(0, 2.35, 0), []);
  const bobRef = useRef();
  const rodRef = useRef();
  const bobRadius = 0.19;
  const currentBob = useRef(new THREE.Vector3());

  useFrame(() => {
    const x = Math.sin(angle) * length;
    const y = pivot.y - Math.cos(angle) * length;
    currentBob.current.set(x, y, 0);
    if (bobRef.current) bobRef.current.position.copy(currentBob.current);
    if (rodRef.current) {
      const mid = pivot.clone().add(currentBob.current).multiplyScalar(0.5);
      rodRef.current.position.copy(mid);
      rodRef.current.scale.set(1, Math.max(0.01, pivot.distanceTo(currentBob.current)), 1);
      rodRef.current.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), currentBob.current.clone().sub(pivot).normalize());
    }
    if (bobPositionRef) bobPositionRef.current = { x, y };
  });

  return (
    <group>
      <mesh position={[0, pivot.y, 0]} castShadow>
        <sphereGeometry args={[0.08, 24, 24]} />
        <meshStandardMaterial color="#f4f7ff" metalness={0.7} roughness={0.25} />
      </mesh>
      <mesh ref={rodRef} castShadow>
        <cylinderGeometry args={[0.018, 0.018, 1, 12]} />
        <meshStandardMaterial color="#9bb3d7" metalness={0.6} roughness={0.3} />
      </mesh>
      <mesh ref={bobRef} castShadow>
        <sphereGeometry args={[bobRadius, 32, 32]} />
        <meshStandardMaterial color={dragging ? '#b6a4ff' : '#55d6ff'} metalness={0.72} roughness={0.22} emissive={dragging ? '#38228c' : '#06354b'} emissiveIntensity={0.6} />
      </mesh>
      <mesh position={[0, pivot.y + 0.07, 0]} castShadow>
        <boxGeometry args={[1.6, 0.13, 0.16]} />
        <meshStandardMaterial color="#45516d" metalness={0.5} roughness={0.4} />
      </mesh>
      <mesh position={[0, 1.15, 0]} castShadow>
        <cylinderGeometry args={[0.055, 0.07, 2.35, 20]} />
        <meshStandardMaterial color="#6b7b9c" metalness={0.65} roughness={0.3} />
      </mesh>
      <mesh position={[0, 0.08, 0]} castShadow>
        <cylinderGeometry args={[0.65, 0.65, 0.16, 32]} />
        <meshStandardMaterial color="#27324c" metalness={0.5} roughness={0.4} />
      </mesh>
      <mesh position={[0, pivot.y - length / 2, -0.035]}>
        <boxGeometry args={[0.012, length, 0.012]} />
        <meshBasicMaterial color="#9fb2d5" transparent opacity={0.22} />
      </mesh>
      <Text position={[0.93, 2.35, 0]} fontSize={0.13} color="#cbd7f1" anchorX="left">Pivot</Text>
      <Text position={[-1.35, 1.35, 0]} fontSize={0.13} color="#aab9d8" anchorX="left">L = {length.toFixed(2)} m</Text>
    </group>
  );
}

function LabScene({ angle, length, dragging, bobPositionRef }) {
  return (
    <>
      <color attach="background" args={['#0b1020']} />
      <fog attach="fog" args={['#0b1020', 7, 16]} />
      <PerspectiveCamera makeDefault position={[4.2, 3.5, 7.5]} fov={42} />
      <ambientLight intensity={0.75} />
      <directionalLight position={[4, 7, 5]} intensity={2.1} castShadow shadow-mapSize-width={1024} shadow-mapSize-height={1024} />
      <pointLight position={[-3, 3, 1]} intensity={18} color="#7064ff" distance={9} />
      <pointLight position={[3, 2, -2]} intensity={12} color="#36d8ff" distance={8} />
      <group position={[0, -0.15, 0]}>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]} receiveShadow>
          <planeGeometry args={[12, 12]} />
          <meshStandardMaterial color="#111a2d" roughness={0.8} metalness={0.15} />
        </mesh>
        <gridHelper args={[12, 24, '#263856', '#17243a']} position={[0, 0.005, 0]} />
      </group>
      <PendulumModel angle={angle} length={length} dragging={dragging} bobPositionRef={bobPositionRef} />
      <Environment preset="city" />
      <OrbitControls enablePan={false} minDistance={6} maxDistance={11} minPolarAngle={0.65} maxPolarAngle={1.55} target={[0, 1.35, 0]} />
    </>
  );
}

export default function App() {
  const videoRef = useRef(null);
  const bobPositionRef = useRef({ x: 0, y: 0.75 });
  const lastHandRef = useRef({ left: null, right: null });
  const pinchLatchRef = useRef(false);
  const [hands, setHands] = useState([]);
  const [length, setLength] = useState(1.5);
  const [angle, setAngle] = useState(0.12);
  const [initialAngle, setInitialAngle] = useState(15);
  const [isRunning, setIsRunning] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [oscillations, setOscillations] = useState(0);
  const [measuredPeriod, setMeasuredPeriod] = useState(null);
  const [notice, setNotice] = useState('Start the camera to control the pendulum with your hands.');
  const [showHelp, setShowHelp] = useState(false);
  const [handMode, setHandMode] = useState('right');
  const physicsRef = useRef({ theta: 0.12, omega: 0, last: performance.now(), previousOmega: 0, lastCrossing: null, crossings: 0 });
  const gestureRef = useRef({ bobGrab: false, lengthGrab: false, lastX: 0, lastY: 0, startDistance: 0, startLength: 1.5 });

  const onHands = useCallback((nextHands) => {
    setHands(nextHands);
    const left = nextHands.find(h => h.id.toLowerCase().includes('left')) || null;
    const right = nextHands.find(h => h.id.toLowerCase().includes('right')) || null;
    lastHandRef.current = { left, right };
    const primary = handMode === 'left' ? left : right;
    const secondary = handMode === 'left' ? right : left;
    const g = gestureRef.current;

    // Right (or selected primary) hand grabs the pendulum bob by pinching.
    if (primary?.pinch && !pinchLatchRef.current && !g.lengthGrab) {
      const inGrabZone = primary.x > 0.2 && primary.x < 0.82 && primary.y > 0.25 && primary.y < 0.92;
      if (inGrabZone) {
        g.bobGrab = true;
        g.lastX = primary.x;
        g.lastY = primary.y;
        pinchLatchRef.current = true;
        setIsDragging(true);
        setIsRunning(false);
        setNotice('Bob grabbed — move your hand to set the release angle.');
      }
    }
    if (g.bobGrab && primary) {
      // Map camera movement to a stable 2D interaction plane. Keep the bob on a plausible arc.
      const deltaX = (primary.x - g.lastX) * 2.5;
      const targetAngle = clamp(physicsRef.current.theta + deltaX, -0.85, 0.85);
      physicsRef.current.theta = targetAngle;
      physicsRef.current.omega = 0;
      setAngle(targetAngle);
      g.lastX = primary.x;
      g.lastY = primary.y;
    }
    if (g.bobGrab && !primary?.pinch) {
      g.bobGrab = false;
      pinchLatchRef.current = false;
      setIsDragging(false);
      setIsRunning(true);
      physicsRef.current.last = performance.now();
      setNotice('Released! Watch the pendulum oscillate.');
    } else if (!primary?.pinch) {
      pinchLatchRef.current = false;
    }

    // Secondary hand pinches to change string length.
    if (secondary?.pinch && !g.bobGrab) {
      if (!g.lengthGrab) {
        g.lengthGrab = true;
        g.lastY = secondary.y;
        g.startLength = length;
        setNotice('Length control active — move your hand up or down.');
      } else {
        const change = (g.lastY - secondary.y) * 1.6;
        setLength(prev => clamp(prev + change, 0.8, 2.0));
        g.lastY = secondary.y;
      }
    } else {
      g.lengthGrab = false;
    }
  }, [handMode, length]);

  const { cameraStatus, cameraError, startCamera, stopCamera } = useHandTracking(videoRef, onHands);

  useEffect(() => {
    let raf = 0;
    const tick = (now) => {
      const p = physicsRef.current;
      const dt = Math.min((now - p.last) / 1000, 0.032);
      p.last = now;
      if (isRunning && !isDragging) {
        // Nonlinear pendulum equation with light damping.
        const damping = 0.08;
        const alpha = -(GRAVITY / length) * Math.sin(p.theta) - damping * p.omega;
        p.omega += alpha * dt;
        p.theta += p.omega * dt;
        if (Math.abs(p.theta) > 0.9) { p.theta = Math.sign(p.theta) * 0.9; p.omega *= -0.45; }
        if (p.previousOmega < 0 && p.omega >= 0) {
          p.crossings += 1;
          setOscillations(Math.floor(p.crossings / 2));
          if (p.lastCrossing !== null) {
            const period = (now - p.lastCrossing) / 1000;
            setMeasuredPeriod(Number(period.toFixed(2)));
          }
          p.lastCrossing = now;
        }
        p.previousOmega = p.omega;
        setAngle(p.theta);
        setElapsed(v => v + dt);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isRunning, isDragging, length]);

  const theoreticalPeriod = 2 * Math.PI * Math.sqrt(length / GRAVITY);

  const resetExperiment = () => {
    physicsRef.current = { theta: initialAngle * Math.PI / 180, omega: 0, last: performance.now(), previousOmega: 0, lastCrossing: null, crossings: 0 };
    gestureRef.current.bobGrab = false;
    gestureRef.current.lengthGrab = false;
    pinchLatchRef.current = false;
    setAngle(initialAngle * Math.PI / 180);
    setIsRunning(false);
    setIsDragging(false);
    setElapsed(0);
    setOscillations(0);
    setMeasuredPeriod(null);
    setNotice('Experiment reset. Pinch the bob to begin.');
  };

  const beginExperiment = () => {
    physicsRef.current.theta = initialAngle * Math.PI / 180;
    physicsRef.current.omega = 0;
    physicsRef.current.last = performance.now();
    setAngle(physicsRef.current.theta);
    setIsRunning(true);
    setNotice('Experiment running — grab the bob to change its angle.');
  };

  const toggleCamera = () => cameraStatus === 'active' ? stopCamera() : startCamera();

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Activity size={22} /></div>
          <div><div className="brand-title">NEXORA <span>LAB</span></div><div className="brand-subtitle">INTERACTIVE PHYSICS STUDIO</div></div>
        </div>
        <div className="topbar-right">
          <span className={`status-pill ${cameraStatus === 'active' ? 'status-live' : ''}`}><span className="status-dot" />{cameraStatus === 'active' ? 'HAND TRACKING LIVE' : cameraStatus === 'loading' ? 'STARTING CAMERA' : 'CAMERA OFF'}</span>
          <button className="icon-button" onClick={() => setShowHelp(v => !v)} aria-label="Toggle instructions"><Info size={18} /></button>
        </div>
      </header>

      <section className="hero-row">
        <div>
          <div className="eyebrow"><GraduationCap size={15} /> CLASS 6–10 · PHYSICS PRACTICAL</div>
          <h1>Simple <span>Pendulum</span></h1>
          <p className="hero-copy">Move beyond the screen. Use your hands to explore motion, gravity and time.</p>
        </div>
        <div className="experiment-chip"><span className="chip-number">01</span><div><strong>MECHANICS</strong><small>Experiment 01 / 08</small></div></div>
      </section>

      {showHelp && <section className="help-banner"><Info size={18} /><div><strong>How to interact</strong><p>Allow camera access. Pinch thumb and index finger on your selected hand to grab/release the bob. Pinch with the other hand and move vertically to adjust string length. You can also use the on-screen controls if camera tracking is unavailable.</p></div></section>}

      <section className="workspace">
        <div className="lab-card scene-card">
          <div className="card-heading"><div><span className="section-kicker">LIVE SIMULATION</span><h2>Virtual workbench</h2></div><span className="live-tag"><span /> 3D SCENE</span></div>
          <div className="scene-wrap">
            <Canvas shadows dpr={[1, 1.7]}>
              <Suspense fallback={null}>
                <LabScene angle={angle} length={length} dragging={isDragging} bobPositionRef={bobPositionRef} />
              </Suspense>
            </Canvas>
            <div className="scene-overlay top-left"><Move3D size={15} /><span>3D INTERACTIVE SPACE</span></div>
            <div className="scene-overlay bottom-left"><span className="legend-dot cyan" /> Pendulum bob <span className="legend-dot violet" /> Active grab</div>
            <div className="angle-readout"><small>LIVE ANGLE</small><strong>{Math.abs(angle * 180 / Math.PI).toFixed(1)}°</strong></div>
            {isDragging && <div className="grab-toast"><Hand size={15} /> BOB GRABBED</div>}
          </div>
          <div className="scene-controls">
            <button className="primary-button" onClick={isRunning ? () => { setIsRunning(false); setNotice('Simulation paused.'); } : beginExperiment}>{isRunning ? <Pause size={17} /> : <Play size={17} />}{isRunning ? 'Pause simulation' : 'Start simulation'}</button>
            <button className="secondary-button" onClick={resetExperiment}><RotateCcw size={16} /> Reset</button>
            <div className="control-spacer" />
            <label className="compact-select">Primary hand
              <select value={handMode} onChange={e => setHandMode(e.target.value)}><option value="right">Right</option><option value="left">Left</option></select>
            </label>
          </div>
        </div>

        <aside className="side-column">
          <section className="lab-card camera-card">
            <div className="card-heading compact"><div><span className="section-kicker">MOTION INPUT</span><h2>Hand tracking</h2></div><span className={`camera-indicator ${cameraStatus}`} /></div>
            <div className="camera-preview">
              <video ref={videoRef} autoPlay playsInline muted className={cameraStatus === 'active' ? 'video-active' : ''} />
              {cameraStatus !== 'active' && <div className="camera-placeholder"><div className="camera-icon"><Camera size={25} /></div><strong>{cameraStatus === 'loading' ? 'Connecting camera…' : 'Camera is ready'}</strong><span>{cameraStatus === 'error' ? 'Check browser permissions' : 'Enable webcam to use hand gestures'}</span></div>}
              {cameraStatus === 'active' && <div className="camera-live-label"><span /> LIVE</div>}
            </div>
            {cameraError && <div className="error-message"><AlertCircle size={15} />{cameraError}</div>}
            <button className={cameraStatus === 'active' ? 'secondary-button full-width' : 'primary-button full-width'} onClick={toggleCamera} disabled={cameraStatus === 'loading'}>
              {cameraStatus === 'active' ? <CameraOff size={17} /> : <Camera size={17} />}{cameraStatus === 'active' ? 'Turn camera off' : cameraStatus === 'loading' ? 'Starting…' : 'Enable webcam'}
            </button>
            <div className="hands-status">
              {['Left', 'Right'].map(name => {
                const hand = hands.find(h => h.id.toLowerCase().includes(name.toLowerCase()));
                return <div className="hand-status-row" key={name}><div className={`hand-icon ${hand ? 'detected' : ''}`}><Hand size={17} /></div><div className="hand-status-copy"><strong>{name} hand</strong><span>{hand ? (hand.pinch ? 'Pinch detected' : 'Tracking') : 'Not detected'}</span></div><span className={`tiny-status ${hand ? 'on' : ''}`} /></div>;
              })}
            </div>
          </section>

          <section className="lab-card controls-card">
            <div className="card-heading compact"><div><span className="section-kicker">EXPERIMENT SETTINGS</span><h2>Adjust parameters</h2></div><Ruler size={19} className="muted-icon" /></div>
            <div className="parameter-block"><div className="parameter-label"><span>String length</span><strong>{length.toFixed(2)} <small>m</small></strong></div><input aria-label="String length" type="range" min="0.8" max="2" step="0.05" value={length} onChange={e => setLength(Number(e.target.value))} /><div className="range-labels"><span>0.8 m</span><span>2.0 m</span></div></div>
            <div className="parameter-block"><div className="parameter-label"><span>Initial angle</span><strong>{initialAngle}<small>°</small></strong></div><input aria-label="Initial angle" type="range" min="5" max="40" step="1" value={initialAngle} onChange={e => { setInitialAngle(Number(e.target.value)); if (!isRunning && !isDragging) setAngle(Number(e.target.value) * Math.PI / 180); }} /><div className="range-labels"><span>5°</span><span>40°</span></div></div>
            <div className="hint-box"><Hand size={17} /><span><strong>Gesture tip</strong> Pinch with your primary hand to grab the bob. Use your other hand to change string length.</span></div>
          </section>
        </aside>
      </section>

      <section className="metrics-grid">
        <div className="metric-card"><div className="metric-icon cyan-bg"><Activity size={18} /></div><div className="metric-label">THEORETICAL PERIOD</div><div className="metric-value">{theoreticalPeriod.toFixed(2)} <small>s</small></div><div className="metric-foot">For small-angle approximation</div></div>
        <div className="metric-card"><div className="metric-icon violet-bg"><RotateCcw size={18} /></div><div className="metric-label">OSCILLATIONS</div><div className="metric-value">{oscillations.toString().padStart(2, '0')}</div><div className="metric-foot">Complete cycles detected</div></div>
        <div className="metric-card"><div className="metric-icon blue-bg"><Activity size={18} /></div><div className="metric-label">ELAPSED TIME</div><div className="metric-value">{elapsed.toFixed(1)} <small>s</small></div><div className="metric-foot">While simulation is running</div></div>
        <div className="metric-card"><div className="metric-icon green-bg"><CheckCircle2 size={18} /></div><div className="metric-label">MEASURED PERIOD</div><div className="metric-value">{measuredPeriod == null ? '—' : measuredPeriod.toFixed(2)} <small>{measuredPeriod == null ? '' : 's'}</small></div><div className="metric-foot">Calculated from cycle timing</div></div>
      </section>

      <section className="bottom-grid">
        <div className="lab-card learn-card"><div className="card-heading compact"><div><span className="section-kicker">LEARN THE CONCEPT</span><h2>What are you discovering?</h2></div></div><p>A simple pendulum is a mass suspended from a fixed point that swings under gravity. Change the string length and compare how long each oscillation takes.</p><div className="formula"><span>Time period</span><strong>T = 2π √(L / g)</strong><small>L = length in metres · g ≈ 9.81 m/s²</small></div></div>
        <div className="lab-card task-card"><div className="card-heading compact"><div><span className="section-kicker">YOUR CHALLENGE</span><h2>Try this experiment</h2></div><GraduationCap size={20} className="muted-icon" /></div><ol><li>Set the length to <strong>1.00 m</strong>.</li><li>Release the bob and observe its motion.</li><li>Change the length to <strong>1.80 m</strong>.</li><li>Compare the theoretical time periods.</li></ol><div className="notice-line"><Info size={15} /><span>{notice}</span></div></div>
      </section>
      <footer><div className="footer-brand"><Activity size={16} /> NEXORA LAB</div><span>Built for curious minds · Class 6–10 Physics</span><span>Camera video stays in your browser.</span></footer>
    </main>
  );
}