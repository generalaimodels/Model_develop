from duckduckgo_search import DDGS

with DDGS() as ddgs:
    results = [print(r) for r in ddgs.images("today_birthday famous celebrities", max_results=100)]
    # print(results)


with  