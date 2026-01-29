from chonkie import TokenChunker
#import products_database 
#une fonction qui transforme une format qlq comme text (exp:json)



def chunkie(text: str):
    chunker = TokenChunker(
        chunk_size=200,
        chunk_overlap=50
    )
    chunks = chunker.chunk(text)
    return [chunk.text for chunk in chunks]


#def process_products(products_database):
 #   result = {}
    
 #   for product in products_database:
  #      product_text = product_to_text(product)
   #     chunks = chunkie(product_text)
    #    result[product['id']] = chunks
    
    #return result

#print(process_products(products_database))