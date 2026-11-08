import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, gradeKick } from '../api';
import Modal from './Modal';
import NameField, { rememberName } from './NameField';

const MAX_SECONDS = 15;          // auto-stop (the backend cuts clips at 15 s anyway)
const COUNTDOWN = 3;
const BITRATE = 1_000_000;       // ~1 MB for a 5-8 s clip

type Stage = 'starting' | 'ready' | 'countdown' | 'recording' | 'review' | 'grading' | 'retake' | 'error' | 'no-camera';

/** Chrome/Firefox record webm, Safari records mp4. */
function pickFormat(): { mimeType: string; ext: string } {
  const options = [
    { mimeType: 'video/webm;codecs=vp9', ext: 'webm' },
    { mimeType: 'video/webm;codecs=vp8', ext: 'webm' },
    { mimeType: 'video/webm', ext: 'webm' },
    { mimeType: 'video/mp4', ext: 'mp4' },
  ];
  return options.find((o) => typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(o.mimeType))
    ?? { mimeType: '', ext: 'webm' };
}

/**
 * + NEW KICK: camera -> record -> watch it back (retake if needed) -> player name -> submit -> grade.
 * Closing at any point turns the camera off.
 */
export default function RecordFlow({ onClose, onGraded }: {
  onClose: () => void;
  onGraded: (resultId: string) => void;
}) {
  const [stage, setStage] = useState<Stage>('starting');
  const [count, setCount] = useState(COUNTDOWN);
  const [elapsed, setElapsed] = useState(0);
  const [clip, setClip] = useState<{ blob: Blob; url: string; ext: string } | null>(null);
  const [player, setPlayer] = useState<string | null>(null);
  const [message, setMessage] = useState('');

  const previewRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const timersRef = useRef<number[]>([]);
  const clipUrlRef = useRef<string | null>(null);

  const clearTimers = () => {
    timersRef.current.forEach((t) => { clearInterval(t); clearTimeout(t); });
    timersRef.current = [];
  };

  // Camera on when the panel opens; everything off when it closes.
  useEffect(() => {
    let cancelled = false;
    navigator.mediaDevices?.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (previewRef.current) previewRef.current.srcObject = stream;
        setStage('ready');
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setMessage(
          e.name === 'NotAllowedError'
            ? 'Camera permission was denied. Allow camera access for this site and try again.'
            : e.name === 'NotReadableError'
              ? 'The camera is in use by another app (Zoom, FaceTime…). Close it and try again.'
              : `Couldn't start the camera: ${e.message}`,
        );
        setStage('no-camera');
      });
    if (!navigator.mediaDevices) {
      setMessage('Camera needs a secure page: open the app on localhost or https.');
      setStage('no-camera');
    }
    return () => {
      cancelled = true;
      clearTimers();
      if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (clipUrlRef.current) URL.revokeObjectURL(clipUrlRef.current);
    };
  }, []);

  // Re-attach the live preview when coming back from review.
  useEffect(() => {
    if ((stage === 'ready' || stage === 'countdown' || stage === 'recording') && previewRef.current && streamRef.current) {
      if (previewRef.current.srcObject !== streamRef.current) previewRef.current.srcObject = streamRef.current;
    }
  }, [stage]);

  const stopRecording = useCallback(() => {
    clearTimers();
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
  }, []);

  function startCountdown() {
    setCount(COUNTDOWN);
    setStage('countdown');
    let n = COUNTDOWN;
    const t = window.setInterval(() => {
      n -= 1;
      setCount(n);
      if (n <= 0) {
        clearInterval(t);
        startRecording();
      }
    }, 1000);
    timersRef.current.push(t);
  }

  function startRecording() {
    const stream = streamRef.current;
    if (!stream) return;
    const { mimeType, ext } = pickFormat();
    const chunks: Blob[] = [];
    const rec = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: BITRATE } : undefined);
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      const blob = new Blob(chunks, { type: rec.mimeType || mimeType || 'video/webm' });
      if (clipUrlRef.current) URL.revokeObjectURL(clipUrlRef.current);
      const url = URL.createObjectURL(blob);
      clipUrlRef.current = url;
      setClip({ blob, url, ext });
      setStage('review');
    };
    recorderRef.current = rec;
    rec.start(250);
    setElapsed(0);
    setStage('recording');
    const started = Date.now();
    const tick = window.setInterval(() => setElapsed((Date.now() - started) / 1000), 100);
    const auto = window.setTimeout(stopRecording, MAX_SECONDS * 1000);
    timersRef.current.push(tick, auto);
  }

  async function submit() {
    if (!clip || !player) return;
    setStage('grading');
    setElapsed(0);
    const started = Date.now();
    const tick = window.setInterval(() => setElapsed((Date.now() - started) / 1000), 250);
    timersRef.current.push(tick);
    try {
      const result = await gradeKick(clip.blob, clip.ext, player);
      clearTimers();
      rememberName(player);
      if (result.status === 'graded') {
        onGraded(result.id);
      } else {
        setMessage(result.retake_reason ?? "The kick couldn't be graded.");
        setStage('retake');
      }
    } catch (e) {
      clearTimers();
      setMessage(e instanceof ApiError ? e.message : String(e));
      setStage('error');
    }
  }

  const live = stage === 'starting' || stage === 'ready' || stage === 'countdown' || stage === 'recording';
  const onName = useCallback((n: string | null) => setPlayer(n), []);

  return (
    <Modal title="New kick" onClose={onClose} closable={stage !== 'grading'}>
      <div className="record-flow">
        {live && (
          <div className="camera">
            {/* Mirrored like a mirror; the recording itself is not mirrored. */}
            <video ref={previewRef} className="camera-preview" autoPlay muted playsInline />
            {stage === 'starting' && <div className="camera-overlay">Starting camera…</div>}
            {stage === 'countdown' && <div className="camera-overlay big">{count}</div>}
            {stage === 'recording' && (
              <div className="rec-indicator">● REC {elapsed.toFixed(1)}s / {MAX_SECONDS}s</div>
            )}
          </div>
        )}

        {stage === 'ready' && (
          <>
            <p className="hint">Stand side-on, whole body in frame, kicking leg closest to the camera.
              Hold still, kick once, hold still again.</p>
            <button className="btn btn-primary btn-wide" onClick={startCountdown}>● RECORD</button>
          </>
        )}
        {stage === 'countdown' && <p className="hint center">Get ready…</p>}
        {stage === 'recording' && (
          <button className="btn btn-danger btn-wide" onClick={stopRecording}>■ STOP</button>
        )}

        {clip && (stage === 'review' || stage === 'grading' || stage === 'error') && (
          <>
            <video className="camera-playback" src={clip.url} controls playsInline autoPlay loop muted />
            {stage === 'review' && (
              <>
                <button className="btn btn-ghost" onClick={() => setStage('ready')}>↺ Retake</button>
                <NameField onReady={onName} />
                <button className="btn btn-primary btn-wide" disabled={!player} onClick={submit}>
                  SUBMIT FOR GRADING
                </button>
              </>
            )}
            {stage === 'grading' && (
              <div className="grading">
                <div className="spinner" aria-hidden="true" />
                <p>Grading… {elapsed.toFixed(0)}s</p>
                <p className="muted">Finding your chamber and extension, then the coach reviews it (10-30 s).</p>
              </div>
            )}
            {stage === 'error' && (
              <div className="notice notice-error">
                {message}
                <div className="row">
                  <button className="btn btn-primary" onClick={() => setStage('review')}>Try again</button>
                </div>
              </div>
            )}
          </>
        )}

        {stage === 'retake' && (
          <div className="notice notice-retry">
            <strong>RETRY</strong> {message}
            <div className="row">
              <button className="btn btn-primary" onClick={() => { setClip(null); setStage('ready'); }}>
                ● Record again
              </button>
            </div>
          </div>
        )}

        {stage === 'no-camera' && <div className="notice notice-error">{message}</div>}
      </div>
    </Modal>
  );
}
