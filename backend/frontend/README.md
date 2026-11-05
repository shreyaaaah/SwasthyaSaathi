# SwasthyaSaathi frontend

React 19 + Vite chat UI for the SwasthyaSaathi FastAPI service.

## Development

```powershell
npm install
$env:VITE_API_BASE_URL = "http://localhost:8000"
npm run dev
```

During local development, the UI defaults to `http://localhost:8000`; set `VITE_API_BASE_URL` to use a different local API. Production builds use the confirmed Render API at `https://swasthyasaathi-3.onrender.com`. Use `npm run build` to produce the production bundle in `dist/` and `npm run preview` to serve it locally.

Voice input uses the browser's speech recognition API when available. The browser may send audio to its configured speech service.
