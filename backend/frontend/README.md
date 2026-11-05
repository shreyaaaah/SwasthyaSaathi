# SwasthyaSaathi frontend

React 19 + Vite chat UI for the SwasthyaSaathi FastAPI service.

## Development

```powershell
npm install
$env:VITE_API_BASE_URL = "http://localhost:8000"
npm run dev
```

During local development, the UI defaults to `http://localhost:8000`. Production builds default to the Render API at `https://swasthya-saathi-3.onrender.com` if `VITE_API_BASE_URL` is not set. Configure `VITE_API_BASE_URL` in Vercel when using a different backend. Use `npm run build` to produce the production bundle in `dist/` and `npm run preview` to serve it locally.

Voice input uses the browser's speech recognition API when available. The browser may send audio to its configured speech service.
