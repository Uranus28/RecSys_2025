import scrapy
import json
from scrapy import Spider, Request
from scrapy_selenium import SeleniumRequest

class artistLinkSpider(Spider):
    name = 'artistLinkSpider'
    host = 'https://www.albumoftheyear.org'
    allowed_domains = ['albumoftheyear.org']
        # start_urls = ['https://www.albumoftheyear.org/releases/this-week/',
        #          'https://www.albumoftheyear.org/releases/this-week/2/'
        #          ]

    start_urls = ['https://www.albumoftheyear.org']
    # headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1"}
    
    async def start(self):
        for url in self.start_urls:
            # yield Request(url=url, headers={self.headers}, callback=self.parse)
            yield SeleniumRequest(url=url, callback=self.parse)

    def parse(self, response):
        pag = response.url.split('/')[-1]
        albumReleases = response.xpath('.//*[@id = "homeNewReleases"]')[0]
        print(albumReleases)
        print("-------------------------------------")
        albumDivs = albumReleases.xpath('.//*[@class = "albumBlock"]')
        found_links = []
        for album in albumDivs:
            artist_Link_block = album.xpath('.//a')[1]
            artistTitle =  album.xpath('.//*[@class ="artistTitle"]/text()')[0].extract()
            albumTitle = album.xpath('.//*[@class ="albumTitle"]/text()')[0].extract()
            link = artist_Link_block.xpath('./@href')[0]
            print(link)
            fullLink = self.host + str(link)
            found_links.append({"link": fullLink, "artistTitle": str(artistTitle), "albumTitle": albumTitle})
        self.writeJSON(f'linksP_{pag}.json', found_links)
        

    def writeJSON(self, fileName, data):
        with open(fileName, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    