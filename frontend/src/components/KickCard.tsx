import { mediaUrl, type Kick } from '../api';
import Stars from './Stars';
import { timeAgo } from '../time';

/** One kick in the feed. Shows the extension frame as a thumbnail; the video only loads in the detail view. */
export default function KickCard({ kick, onOpen, onPlayer }: {
  kick: Kick;
  onOpen: () => void;
  onPlayer: (username: string) => void;
}) {
  const thumb = mediaUrl(kick.images.extension ?? kick.images.chamber);
  return (
    <article className={`kick-card status-${kick.status}`}>
      <button className="kick-card-main" onClick={onOpen} aria-label={`Open kick by ${kick.username}`}>
        <div className="thumb">
          {thumb ? <img src={thumb} alt="" loading="lazy" /> : <span className="thumb-empty">NO IMAGE</span>}
        </div>
        <div className="kick-card-info">
          <div className="kick-card-score">
            {kick.status === 'completed' && kick.passed !== null ? (
              <>
                <Stars passed={kick.passed} />
                <span className={`tier tier-${kick.tier}`}>{kick.tier}</span>
              </>
            ) : kick.status === 'failed' ? (
              <span className="badge badge-retry">RETRY</span>
            ) : (
              <span className="badge badge-busy">GRADING…</span>
            )}
          </div>
          <div className="kick-card-time">{timeAgo(kick.recorded_time)}</div>
        </div>
      </button>
      <button className="player-link" onClick={() => onPlayer(kick.username)} title={`See all kicks by ${kick.username}`}>
        @{kick.username}
      </button>
    </article>
  );
}
