# Mobile web (React + TypeScript)

## Develop
```bash
cd mobile-web
npm install
npm run dev
```
Open the Vite URL (default `http://localhost:5173/app/`). Point **Settings → API URL** at your local-api (`http://<pc-ip>:8001`).

## Production (served by local-api)
```bash
cd mobile-web
npm run build
```
local-api serves `mobile-web/dist` at `/app/`.
