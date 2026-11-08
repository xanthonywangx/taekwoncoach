import { useEffect, useState } from 'react';
import { searchUsers } from '../api';

/** Search players by name; picking one switches the feed to their kicks. */
export default function SearchBar({ onPick }: { onPick: (username: string) => void }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<string[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const q = query.trim().toLowerCase();
    if (!q) {
      setResults([]);
      return;
    }
    let cancelled = false;
    const t = setTimeout(() => {
      searchUsers(q)
        .then((rows) => !cancelled && setResults(rows.map((r) => r.username)))
        .catch(() => !cancelled && setResults([]));
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [query]);

  function pick(name: string) {
    onPick(name);
    setQuery('');
    setOpen(false);
  }

  return (
    <div className="search" onBlur={(e) => !e.currentTarget.contains(e.relatedTarget) && setOpen(false)}>
      <input
        type="search"
        placeholder="Search players…"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => e.key === 'Enter' && results[0] && pick(results[0])}
        aria-label="Search players"
      />
      {open && query.trim() && (
        <ul className="search-results" role="listbox">
          {results.length === 0 ? (
            <li className="muted">No players found</li>
          ) : (
            results.map((name) => (
              <li key={name}>
                <button onClick={() => pick(name)}>@{name}</button>
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}
