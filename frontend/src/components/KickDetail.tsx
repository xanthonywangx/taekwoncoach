import { useState } from 'react';
import { ApiError, CHECK_LABELS, deleteKick, mediaUrl, type CheckName, type Kick } from '../api';
import Stars from './Stars';
import { timeAgo } from '../time';

const MEASURE_LABELS: Record<string, (v: number) => string> = {
  ankle_forward_of_knee_ratio: (v) => `foot ${v > 0 ? 'ahead of' : 'behind'} knee (${v.toFixed(2)})`,
  knee_below_hip_ratio: (v) => `knee ${v > 0 ? 'below' : 'above'} hip (${v.toFixed(2)})`,
  knee_angle_deg: (v) => `knee angle ${v.toFixed(0)}°`,
  leg_angle_deg: (v) => `leg angle ${v.toFixed(0)}°`,
};

/** The focused kick: video, the two graded frames, the 4 checks, stars, feedback, delete. */
export default function KickDetail({ kick, onDeleted }: { kick: Kick; onDeleted: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const clip = mediaUrl(kick.clip_url);
  const graded = kick.status === 'completed' && kick.checks && kick.passed !== null;

  async function onDelete() {
    setBusy(true);
    setError(null);
    try {
      await deleteKick(kick.result_id, name);
      onDeleted();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="kick-detail">
      <div className="detail-meta">
        <span className="player">@{kick.username}</span>
        <span className="muted">{timeAgo(kick.recorded_time)}</span>
      </div>

      {graded ? (
        <div className="detail-score">
          <Stars passed={kick.passed!} size="lg" />
          <span className={`tier tier-${kick.tier}`}>{kick.tier}</span>
        </div>
      ) : kick.status === 'failed' ? (
        <div className="notice notice-retry">
          <strong>RETRY</strong> {kick.fail_reason || 'This kick could not be graded.'}
        </div>
      ) : (
        <div className="notice">Still grading…</div>
      )}

      {clip && (
        // The video only loads here, never in the feed (each clip is ~1 MB).
        <video className="detail-video" src={clip} controls playsInline preload="metadata" />
      )}

      {(kick.images.chamber || kick.images.extension) && (
        <div className="frames">
          {kick.images.chamber && (
            <figure>
              <img src={mediaUrl(kick.images.chamber)} alt="Chamber frame" />
              <figcaption>Chamber</figcaption>
            </figure>
          )}
          {kick.images.extension && (
            <figure>
              <img src={mediaUrl(kick.images.extension)} alt="Extension frame" />
              <figcaption>Extension</figcaption>
            </figure>
          )}
        </div>
      )}

      {graded && (
        <ul className="checks">
          {(Object.keys(CHECK_LABELS) as CheckName[]).map((name) => {
            const c = kick.checks![name];
            const measure = Object.entries(c).find(([k]) => k !== 'pass');
            return (
              <li key={name} className={c.pass ? 'check pass' : 'check fail'}>
                <span className="check-mark">{c.pass ? '✔' : '✖'}</span>
                <span className="check-name">{CHECK_LABELS[name]}</span>
                {measure && MEASURE_LABELS[measure[0]] && (
                  <span className="check-value muted">{MEASURE_LABELS[measure[0]](measure[1])}</span>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {kick.feedback && (
        <div className="feedback">
          <div className="feedback-label">COACH {kick.feedback_source === 'nemotron' ? '· NEMOTRON' : ''}</div>
          <p>{kick.feedback}</p>
        </div>
      )}

      <div className="danger-zone">
        {!confirming ? (
          <button className="btn btn-ghost" onClick={() => setConfirming(true)}>
            Delete kick
          </button>
        ) : (
          <div className="confirm-delete">
            <label htmlFor="del-name">Type the player name to delete</label>
            <div className="row">
              <input
                id="del-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={kick.username}
                autoComplete="off"
                autoFocus
              />
              <button className="btn btn-danger" disabled={!name.trim() || busy} onClick={onDelete}>
                {busy ? 'Deleting…' : 'Delete'}
              </button>
              <button className="btn btn-ghost" onClick={() => { setConfirming(false); setError(null); }}>
                Cancel
              </button>
            </div>
            {error && <p className="error">{error}</p>}
          </div>
        )}
      </div>
    </div>
  );
}
