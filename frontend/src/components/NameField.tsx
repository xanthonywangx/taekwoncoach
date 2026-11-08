import { useEffect, useState } from 'react';
import { ApiError, checkUsername, createPlayer, normalizeName, USERNAME_RE } from '../api';

export type NameState = 'empty' | 'invalid' | 'checking' | 'exists' | 'new' | 'creating' | 'error';

const LAST_NAME_KEY = 'taekwoncoach.lastPlayer';

export function rememberName(name: string) {
  try {
    localStorage.setItem(LAST_NAME_KEY, name);
  } catch {
    /* private mode etc. */
  }
}

function lastName() {
  try {
    return localStorage.getItem(LAST_NAME_KEY) ?? '';
  } catch {
    return '';
  }
}

/**
 * Player name for the kick. Checks the name as you type:
 *   existing player -> "Welcome back", ready to submit
 *   new name        -> "Create player?" button (so a typo doesn't silently make a new player)
 */
export default function NameField({ onReady }: { onReady: (name: string | null) => void }) {
  const [raw, setRaw] = useState(lastName);
  const [state, setState] = useState<NameState>('empty');
  const [message, setMessage] = useState('');
  const name = normalizeName(raw);

  useEffect(() => {
    if (!name) {
      setState('empty');
      onReady(null);
      return;
    }
    if (!USERNAME_RE.test(name)) {
      setState('invalid');
      onReady(null);
      return;
    }
    setState('checking');
    onReady(null);
    let cancelled = false;
    const t = setTimeout(() => {
      checkUsername(name)
        .then(() => {
          if (cancelled) return;
          setState('exists');
          onReady(name);
        })
        .catch((e) => {
          if (cancelled) return;
          if (e instanceof ApiError && e.status === 404) {
            setState('new');
          } else {
            setState('error');
            setMessage(e instanceof Error ? e.message : String(e));
          }
        });
    }, 350);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [name, onReady]);

  async function create() {
    setState('creating');
    try {
      await createPlayer(name);
      setState('exists');
      onReady(name);
    } catch (e) {
      setState('error');
      setMessage(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="name-field">
      <label htmlFor="player-name">PLAYER NAME</label>
      <input
        id="player-name"
        value={raw}
        onChange={(e) => setRaw(e.target.value)}
        placeholder="e.g. haris"
        autoComplete="off"
        autoCapitalize="none"
        spellCheck={false}
        maxLength={20}
        autoFocus
      />
      <div className={`name-status name-${state}`} aria-live="polite">
        {state === 'invalid' && '3-20 characters: letters, numbers or _'}
        {state === 'checking' && 'Checking…'}
        {state === 'exists' && `Welcome back, ${name}!`}
        {state === 'new' && (
          <>
            New player <strong>{name}</strong>.{' '}
            <button className="btn btn-small" onClick={create}>Create player</button>
          </>
        )}
        {state === 'creating' && 'Creating player…'}
        {state === 'error' && <span className="error">{message}</span>}
      </div>
    </div>
  );
}
