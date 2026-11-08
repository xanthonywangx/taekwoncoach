import { useCallback, useEffect, useState } from 'react';
import { ApiError, getRecentKicks, getUserKicks, type Kick } from './api';
import KickCard from './components/KickCard';
import KickDetail from './components/KickDetail';
import Modal from './components/Modal';
import RecordFlow from './components/RecordFlow';
import SearchBar from './components/SearchBar';
import Sky from './components/Sky';

/**
 * One scrolling page:
 *   top bar (title, search, + button) -> feed of kicks (everyone, or one player after a search)
 *   click a card -> detail modal;  + -> record modal (record, review, name, submit, grade)
 */
export default function App() {
  const [player, setPlayer] = useState<string | null>(null); // null = everyone
  const [kicks, setKicks] = useState<Kick[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [focused, setFocused] = useState<Kick | null>(null);
  const [recording, setRecording] = useState(false);

  const load = useCallback(async (who: string | null): Promise<Kick[]> => {
    setLoading(true);
    setError(null);
    try {
      const rows = who ? await getUserKicks(who) : await getRecentKicks();
      setKicks(rows);
      return rows;
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setKicks([]);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(player);
  }, [player, load]);

  // A kick was graded: refresh the feed and open the new kick so its grade is front and center.
  async function onGraded(resultId: string) {
    setRecording(false);
    const rows = await load(player);
    const fresh = rows.find((k) => k.result_id === resultId)
      ?? (await getRecentKicks(20).catch(() => [])).find((k) => k.result_id === resultId);
    if (fresh) setFocused(fresh);
  }

  return (
    <>
      <Sky />

      <header className="topbar">
        <h1 className="logo" onClick={() => setPlayer(null)}>TAEKWONCOACH</h1>
        <SearchBar onPick={setPlayer} />
      </header>

      <main className="feed-wrap">
        <div className="feed-heading">
          {player ? (
            <>
              <h2>@{player}</h2>
              <button className="btn btn-ghost" onClick={() => setPlayer(null)}>◀ All players</button>
            </>
          ) : (
            <h2>Latest kicks</h2>
          )}
        </div>

        {error && (
          <div className="notice notice-error">
            {error} <button className="btn btn-ghost" onClick={() => load(player)}>Retry</button>
          </div>
        )}

        {loading && kicks.length === 0 ? (
          <p className="muted center">Loading…</p>
        ) : !error && kicks.length === 0 ? (
          <p className="muted center">
            {player ? `@${player} hasn't recorded a kick yet.` : 'No kicks yet. Hit the + button to record the first one.'}
          </p>
        ) : (
          <section className="feed">
            {kicks.map((k) => (
              <KickCard key={k.result_id} kick={k} onOpen={() => setFocused(k)} onPlayer={setPlayer} />
            ))}
          </section>
        )}
      </main>

      {/* Floating + button: stays in the bottom-right corner while you scroll. */}
      <button className="fab" onClick={() => setRecording(true)} aria-label="Record a new kick" title="Record a new kick">
        <svg viewBox="0 0 7 7" shapeRendering="crispEdges" aria-hidden="true">
          <rect x="3" y="0" width="1" height="7" fill="currentColor" />
          <rect x="0" y="3" width="7" height="1" fill="currentColor" />
        </svg>
      </button>

      {focused && (
        <Modal title="Kick" onClose={() => setFocused(null)}>
          <KickDetail
            kick={focused}
            onDeleted={() => {
              setFocused(null);
              load(player);
            }}
          />
        </Modal>
      )}

      {recording && <RecordFlow onClose={() => setRecording(false)} onGraded={onGraded} />}
    </>
  );
}
