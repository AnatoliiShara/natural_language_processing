from typing import * 
import xml.sax 

class SynonymHandler(xml.sax.ContentHandler):
	
	def __init__(self):
		self.current_tag = ""
		self.synonyms: List[str] = []
	
	def startElements(self, tag: str, attributes):
		self.current_tag = tag

	def endElement(self, tag: str):
		self.current_tag = ""

	def characters(self, content: str):
		if self.current_tag == "synonym":
			self.synonyms.append(content)


def process_wiktionary_dump(dump_url: str) -> List[str]:
	parser = xml.sax.make_parser()
	handler = SynonymHandler()
	parser.setContentHandler(handler)
	with open(dump_url, "r", encoding='utf-8') as f:
		xml.sax.parse(f, handler)

def process_plain_text_file(file_path: str) -> List[str]:
    synonyms = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Process each line as needed
            synonyms.append(line.strip())  # Assuming each line contains a synonym
            
    return synonyms

# Example usage
plain_text_file_path = "wikidatawiki-20230801-pages-articles-multistream-index27.txt-p110685874p112185873"
synonyms = process_plain_text_file(plain_text_file_path)
print(synonyms[:10])  # Print first 10 lines (synonyms)




import requests

def get_wikidata_item_info(item_id):
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": item_id,
        "format": "json"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "entities" in data and item_id in data["entities"]:
        return data["entities"][item_id]
    else:
        return None

# Example usage
item_ids = ['Q115978282', 'Q115978283', 'Q115978284']
for item_id in item_ids:
    item_info = get_wikidata_item_info(item_id)
    if item_info:
        print(f"Item ID: {item_id}")
        print(f"Label: {item_info.get('labels', {}).get('en', {}).get('value')}")
        print(f"Description: {item_info.get('descriptions', {}).get('en', {}).get('value')}")
        print(f"Claims: {item_info.get('claims')}")
        print("-" * 50)






