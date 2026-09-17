# How to fill knowledge_base/content/

## Option A — run the scraper (fastest)
```bash
cd knowledge_base
pip install requests beautifulsoup4
python scrape_sources.py
```
This fetches every URL in `sources.json`, strips nav/ads/scripts, and writes
one clean `.md` file per source into `content/`, each already tagged with:

```
TITLE: ...
URL: ...
RETRIEVED: 2026-09-10
---

<article text>
```

Some sites (especially visitsingapore.com, which loads content via
JavaScript) may return little or no text to a plain scraper. The script will
print a WARNING for any file under ~200 characters — check those first.

## Option B — manual paste (if a site blocks scraping or returns JS-rendered content)
1. Open the URL in your browser.
2. Select and copy the main article text (skip menus, footers, cookie banners).
3. Open the matching file already created in `content/` (or create it if
   missing) and paste the text *below* the `---` line — the header is already
   there.
4. Save.

## Checklist
- [ ] `content/wikivoyage_singapore.md`
- [ ] `content/visit_singapore_essentials.md`
- [ ] `content/visit_singapore_itineraries.md`
- [ ] `content/visit_singapore_things_to_do.md`

Once all four files have real content (a few hundred to a few thousand words
each is plenty), move on to `ingest.py` to chunk and embed them into Chroma.

## A note on reuse terms
Wikivoyage content is under CC-BY-SA — you can reuse it with attribution
(which `sources.json` already provides). Visit Singapore's site has its own
terms of use; for a course/portfolio assignment this is standard fair use of
a small excerpt for RAG, but don't redistribute the scraped files publicly
as a standalone dataset.
