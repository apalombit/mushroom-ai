"""
Wikipedia source fetcher.

Fetches species pages via the Wikipedia API (CC-BY-SA licensed).
Returns raw text content for each species to be parsed by the extraction LLM.

TODO:
    - [ ] Implement fetch_species_page() using Wikipedia REST API
    - [ ] Handle redirects (common names → scientific names)
    - [ ] Handle missing pages gracefully
    - [ ] Rate limiting to be respectful to Wikipedia
    - [ ] Cache fetched pages locally to avoid re-fetching
"""

SOURCE_NAME = "Wikipedia"
BASE_URL = "https://en.wikipedia.org/w/api.php"
