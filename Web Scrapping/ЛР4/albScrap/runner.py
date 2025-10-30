import scrapy
import json

from scrapy.crawler import CrawlerProcess
# from scrapy.settings import Settings
# from albScrap.settings import settings
from scrapy.utils.project import get_project_settings
from albScrap.spiders.artistLinkSpider import artistLinkSpider # Замените на имя вашего паука

from pymongo import MongoClient
from pymongo import ReturnDocument

def run_spider() -> None:
    """Запускает паука Scrapy из Python кода."""
    process = CrawlerProcess(get_project_settings())
    process.crawl(artistLinkSpider)

    process.start()

def insert_update_one(collection, new_data):
    link = new_data['link']
    filter_query = {'link': link}
    set_new_values = {'$setOnInsert': new_data }
    # Используем update_one с upsert=True и $setOnInsert
    result = collection.update_one(
        filter_query,
        set_new_values,
        upsert=True
    )

    if result.upserted_id:
        print(f"СТрока с link {link} был создан.")
    else:
        print("СТрока с таким url уже существует(не обновлен).")

def write_to_mongodb():
    # Making Connection
    myclient = MongoClient("mongodb://localhost:27017/")
    # database
    db = myclient["dataScrappy"]
    Collection = db["data"]
    newlyReleased_data = {}
    # Loading or Opening the json file
    with open('linksP_www.albumoftheyear.org.json') as file:
        newlyReleased_data = json.load(file)
    for nData in newlyReleased_data:
        insert_update_one(Collection, nData)


if __name__ == "__main__":
    run_spider()
    write_to_mongodb()