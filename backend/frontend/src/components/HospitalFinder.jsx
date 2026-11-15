import React, { useId, useState } from 'react';
import { ExternalLink, Locate, MapPin, Search } from 'lucide-react';

function mapsSearchUrl(place) {
  const url = new URL('https://www.google.com/maps/search/');
  url.searchParams.set('api', '1');
  url.searchParams.set('query', `hospitals near ${place}`);
  return url.toString();
}

function HospitalFinder({ compact = false }) {
  const inputId = useId();
  const [place, setPlace] = useState('');
  const [coordinates, setCoordinates] = useState(null);
  const [locationStatus, setLocationStatus] = useState('');
  const [error, setError] = useState('');
  const [isLocating, setIsLocating] = useState(false);

  const searchPlace = (event) => {
    event.preventDefault();
    const cleanedPlace = place.trim();
    if (!cleanedPlace) return;
    window.open(mapsSearchUrl(cleanedPlace), '_blank', 'noopener,noreferrer');
  };

  const useCurrentLocation = () => {
    setError('');
    setLocationStatus('');
    setCoordinates(null);

    if (!navigator.geolocation) {
      setError('This browser does not support location. Search by area or PIN code instead.');
      return;
    }

    setIsLocating(true);
    navigator.geolocation.getCurrentPosition(
      ({ coords }) => {
        setCoordinates({ latitude: coords.latitude, longitude: coords.longitude });
        setLocationStatus('Location ready. Open Maps to see nearby hospitals.');
        setIsLocating(false);
      },
      (locationError) => {
        const messages = {
          1: 'Location permission was denied. Search by area or PIN code instead.',
          2: 'Your location is unavailable right now. Search by area or PIN code instead.',
          3: 'The location request timed out. Try again or search by area.'
        };
        setError(messages[locationError.code] || 'Could not get your location. Search by area or PIN code instead.');
        setIsLocating(false);
      },
      { enableHighAccuracy: false, maximumAge: 60000, timeout: 12000 }
    );
  };

  const coordinateQuery = coordinates
    ? `${coordinates.latitude.toFixed(5)}, ${coordinates.longitude.toFixed(5)}`
    : '';

  return (
    <section aria-label="Find nearby hospitals" className={`${compact ? '' : 'rounded-2xl border border-slate-700/80 bg-slate-900/70 p-4 shadow-xl'}`}>
      <div className="mb-4">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
          <MapPin aria-hidden="true" className="h-4 w-4 text-teal-300" />
          Nearby hospitals
        </h2>
        <p className="mt-1 text-xs leading-relaxed text-slate-400">Search by city, area, or PIN code, or use your device location.</p>
      </div>

      <form onSubmit={searchPlace} className="space-y-2">
        <label htmlFor={inputId} className="sr-only">City, area, or PIN code</label>
        <input
          id={inputId}
          type="text"
          autoComplete="postal-code"
          maxLength={120}
          value={place}
          onChange={(event) => {
            setPlace(event.target.value);
            setCoordinates(null);
            setLocationStatus('');
            setError('');
          }}
          placeholder="e.g. Indiranagar, Bengaluru"
          className="w-full rounded-lg border border-slate-600 bg-slate-950/80 px-3 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-teal-400 focus:outline-none focus:ring-2 focus:ring-teal-500/25"
        />
        <button
          type="submit"
          disabled={!place.trim()}
          className="flex min-h-10 w-full items-center justify-center gap-2 rounded-lg bg-teal-600 px-3 py-2 text-xs font-semibold text-white transition hover:bg-teal-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Search aria-hidden="true" className="h-3.5 w-3.5" />
          Find hospitals in Maps
          <ExternalLink aria-hidden="true" className="h-3 w-3" />
        </button>
      </form>

      <div className="my-3 flex items-center gap-2 text-[10px] uppercase tracking-wider text-slate-500">
        <span className="h-px flex-1 bg-slate-800" />
        or
        <span className="h-px flex-1 bg-slate-800" />
      </div>

      <button
        type="button"
        onClick={useCurrentLocation}
        disabled={isLocating}
        className="flex min-h-10 w-full items-center justify-center gap-2 rounded-lg border border-slate-600 bg-slate-800/70 px-3 py-2 text-xs font-medium text-slate-200 transition hover:border-teal-500/70 hover:bg-teal-950/40 disabled:cursor-wait disabled:opacity-60"
      >
        <Locate aria-hidden="true" className="h-3.5 w-3.5 text-teal-300" />
        {isLocating ? 'Getting location…' : 'Use my current location'}
      </button>

      {coordinates && (
        <a
          href={mapsSearchUrl(coordinateQuery)}
          target="_blank"
          rel="noreferrer"
          className="mt-2 flex min-h-10 items-center justify-center gap-2 rounded-lg border border-teal-700/70 bg-teal-950/40 px-3 py-2 text-xs font-semibold text-teal-100 hover:bg-teal-900/50"
        >
          Open nearby hospitals <ExternalLink aria-hidden="true" className="h-3 w-3" />
        </a>
      )}

      {locationStatus && <p role="status" className="mt-2 text-[11px] leading-relaxed text-emerald-300">{locationStatus}</p>}
      {error && <p role="alert" className="mt-2 text-[11px] leading-relaxed text-amber-200">{error}</p>}

      <p className="mt-4 text-[10px] leading-relaxed text-slate-500">Your place or coordinates are sent to Google Maps when you search. This app does not save them. Check hospital details before travelling; listings may be incomplete.</p>
      <p className="mt-2 text-[10px] text-slate-500">Map search by Google Maps; no API key required.</p>
    </section>
  );
}

export default HospitalFinder;
