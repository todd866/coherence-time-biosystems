# Coherence Time native paper site

This is a local Next.js clone of the `/ai/75_beyond_bayesian/R1/site`
native-paper interface, with paper-specific TeX, PDF, figures, citation
inventory, and PaperLibrary status.

## Local use

```sh
npm run dev
npm run build
```

The local generator creates `node_modules` as a symlink to the AI paper
site's installed dependencies when possible, so no package fetch is needed
for local builds.
