import React, { useId, useState } from 'react';
import { ExternalLink, Locate, MapPin, Navigation, Search } from 'lucide-react';
import { findNearbyHospitals } from '../api/client';

function mapsPlaceUrl(place) {
  const url = new URL('https://www.google.com/maps/search/');
  url.searchParams.set('api', '1');
  url.searchParams.set('query', place);
  return url.toString();
}

function directionsUrl(latitude, longitude) {
  const url = new URL('https://www.google.com/maps/dir/');
  url.searchParams.set('api', '1');
  url.searchParams.set('destination', `${latitude},${longitude}`);
  return url.toString();
}

function formatDistance(distanceMeters) {
  const distance = Number(distanceMeters);
  if (!Number.isFinite(distance) || distance < 0) return '';
  if (distance < 1000) return `${Math.round(distance)} m away`;
  return `${(distance / 1000).toFixed(1)} km away`;
}

function HospitalFinder({ compact = false }) {
  const inputId = useId();
  const [place, setPlace] = useState('');
  const [coordinates, setCoordinates] = useState(null);
  const [locationStatus, setLocationStatus] = useState('');
  const [error, setError] = useState('');
  const [isLocating, setIsLocating] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState(null);

  const selectedPlace = place.trim();
  const hasLocation = Boolean(selectedPlace || coordinates);

  const openMap = () => {
    const query = coordinates
      ? `${coordinates.latitude},${coordinates.longitude}`
      : selectedPlace;
    if (!query) return;
    window.open(mapsPlaceUrl(query), '_blank', 'noopener,noreferrer');
  };

  const showNearestHospitals = async () => {
    if (!hasLocation) return;
    setError('');
    setSearchResults(null);
    setIsSearching(true);
    try {
      const data = await findNearbyHospitals(coordinates
        ? { latitude: coordinates.latitude, longitude: coordinates.longitude }
        : { place: selectedPlace });
      setSearchResults(data);
    } catch (searchError) {
      setError(searchError.message || 'Could not find nearby hospitals. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  const useCurrentLocation = () => {
    setError('');
    setLocationStatus('');
    setCoordinates(null);
    setSearchResults(null);

    if (!navigator.geolocation) {
      setError('This browser does not support location. Search by area or PIN code instead.');
      return;
    }

    setIsLocating(true);
    navigator.geolocation.getCurrentPosition(
      ({ coords }) => {
        setCoordinates({ latitude: coords.latitude, longitude: coords.longitude });
        setPlace('');
        setLocationStatus('Location ready. Choose Maps or an in-app hospital list.');
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

  return (
    <section aria-label="Find nearby hospitals" className={`${compact ? '' : 'rounded-2xl border border-slate-700/80 bg-slate-900/70 p-4 shadow-xl'}`}>
      {!compact && <div className="mb-4">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
          <MapPin aria-hidden="true" className="h-4 w-4 text-teal-300" />
          Nearby hospitals
        </h2>
        <p className="mt-1 text-xs leading-relaxed text-slate-400">Search by city, area, or PIN code, or use your device location.</p>
      </div>}

      <div className="space-y-2">
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
            setSearchResults(null);
          }}
          placeholder="e.g. Indiranagar, Bengaluru"
          className="w-full rounded-lg border border-slate-600 bg-slate-950/80 px-3 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-teal-400 focus:outline-none focus:ring-2 focus:ring-teal-500/25"
        />
        <button
          type="button"
          onClick={openMap}
          disabled={!hasLocation}
          className="flex min-h-10 w-full items-center justify-center gap-2 rounded-lg border border-slate-600 bg-slate-800/70 px-3 py-2 text-xs font-semibold text-slate-100 transition hover:border-teal-500/70 hover:bg-teal-950/40 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <MapPin aria-hidden="true" className="h-3.5 w-3.5 text-teal-300" />
          {coordinates ? 'Open my location in Maps' : 'Open place in Maps'}
          <ExternalLink aria-hidden="true" className="h-3 w-3" />
        </button>
        <button
          type="button"
          onClick={showNearestHospitals}
          disabled={!hasLocation || isSearching}
          className="flex min-h-10 w-full items-center justify-center gap-2 rounded-lg bg-teal-600 px-3 py-2 text-xs font-semibold text-white transition hover:bg-teal-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Search aria-hidden="true" className="h-3.5 w-3.5" />
          {isSearching ? 'Finding nearby hospitals…' : coordinates ? 'List hospitals near me' : 'List nearest hospitals here'}
        </button>
      </div>

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

      {locationStatus && <p role="status" className="mt-2 text-[11px] leading-relaxed text-emerald-300">{locationStatus}</p>}
      {error && <p role="alert" className="mt-2 text-[11px] leading-relaxed text-amber-200">{error}</p>}

      {searchResults && (
        <div className="mt-4 border-t border-slate-700 pt-3">
          <h3 className="text-xs font-semibold text-slate-100">Hospitals near {searchResults.search_location.label}</h3>
          {searchResults.hospitals.length === 0 ? (
            <p className="mt-2 text-[11px] leading-relaxed text-slate-400">No hospital listings were found within 15 km. Try a nearby area or open Maps to search a wider area.</p>
          ) : (
            <ol className="mt-2 space-y-2">
              {searchResults.hospitals.map((hospital, index) => (
                <li key={`${hospital.latitude}-${hospital.longitude}-${index}`} className="rounded-lg border border-slate-700 bg-slate-950/50 p-2.5">
                  <p className="text-xs font-semibold leading-relaxed text-slate-100">{hospital.name}</p>
                  <p className="mt-0.5 text-[10px] leading-relaxed text-slate-400">{hospital.address}</p>
                  {hospital.distance_meters != null && <p className="mt-1 text-[10px] text-teal-200">{formatDistance(hospital.distance_meters)}</p>}
                  <a
                    href={directionsUrl(hospital.latitude, hospital.longitude)}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-2 inline-flex items-center gap-1 text-[10px] font-semibold text-teal-300 hover:text-teal-200"
                  >
                    <Navigation aria-hidden="true" className="h-3 w-3" /> Directions <ExternalLink aria-hidden="true" className="h-2.5 w-2.5" />
                  </a>
                </li>
              ))}
            </ol>
          )}
          <p className="mt-2 text-[10px] leading-relaxed text-slate-500">Listings are provided by Geoapify and may be incomplete. Confirm details before travelling.</p>
        </div>
      )}

      <p className="mt-4 text-[10px] leading-relaxed text-slate-500">Your location is sent to Geoapify only when you request the in-app list. Opening Maps sends it to Google Maps. This app does not save it.</p>
      <p className="mt-2 text-[10px] text-slate-500">Map search works without a key. The in-app list uses a configured places service.</p>
    </section>
  );
}

export default HospitalFinder;
