import newspaper

corpus = []
titles = []
# Global output from the module
# Use len(corpus_build.corpus) to determine extracted article count

def build_nf(news_feed):
    news_source = newspaper.build(news_feed, memoize_articles=False)
    for article in news_source.articles:
        article.download()
        article.parse()
        if article.text not in corpus:
            titles.append(article.title)
            corpus.append(article.text)

if __name__ == '__main__':
    # Output for a command line module call
    build_nf('https://www.bloomberg.com/view/topics/finance')
    print(corpus)
