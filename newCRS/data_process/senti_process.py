import json
import re
import html
from tqdm.auto import tqdm



movie_pattern = re.compile(r'@\d+')

def process_utt(utt, movieid2name, replace_movieId):
    def convert(match):
        movieid = match.group(0)[1:]
        if movieid in movieid2name:
            movie_name = movieid2name[movieid]
            movie_name = ' '.join(movie_name.split())
            return movie_name
        else:
            return match.group(0)

    if replace_movieId:
        utt = re.sub(movie_pattern, convert, utt)
    utt = ' '.join(utt.split())
    utt = html.unescape(utt)

    return utt

### Original Version

def reformat_data(input_file, output_file, dpath):
    with open(dpath + input_file, 'r', encoding='utf-8') as fin, open(dpath + output_file, 'w', encoding='utf-8') as fout:
        unique_conv_id = 0

        for line in tqdm(fin):
            conversation = json.loads(line)
            dialogs = conversation["dialog"]
            movieid2name = conversation["movieMentions"]
            contexts = []
            entities = []

            for i, message in enumerate(dialogs):
                # @movie_id => movie_name
                utt = process_utt(message["text"], movieid2name, replace_movieId=True)
                #utt = message["text"]
                
                role_prefix = "User: " if message["role"] == "Seeker" else "System: "
                formatted_text = role_prefix + utt #message["text"]

                # Update context and entities
                contexts.append(formatted_text)
                #contexts.append(utt)
                entity_ids = [entity2id[entity] for entity in message["entity"] if entity in entity2id]
                movie_ids = [entity2id[movie] for movie in message["movies"] if movie in entity2id]
                
                entities.extend(entity_ids)
                entities.extend(movie_ids)
                
                if message["role"] == "Seeker":
                    # # 다음 message가 대화의 마지막 message이면서 "role"이 "Recommender" 일 때
                    # if i + 1 == len(dialogs) - 1 and dialogs[i + 1]["role"] == "Recommender":
                    #     next_entities = [entity2id[entity] for entity in dialogs[i + 1]["entity"] if entity in entity2id]
                    #     entities.extend(next_entities)
                        
                    # Write the current context and entities
                    fout.write(json.dumps({
                        "conv_id": conversation["conv_id"],
                        #"unique_conv_id": str(unique_conv_id),
                        "contexts": contexts,
                        "entities": list(set(entities))
                    }, ensure_ascii=False) + '\n')
                    
                    # Prepare for the next context
                    unique_conv_id += 1
                    #context = [formatted_text]  # Start new context with current Seeker's utterance
                    entities = [] # Reset entities for the new Seeker





# ##### Only user utterance (no 누적) (only entity)
# def reformat_data(input_file, output_file, dpath):
#     with open(dpath + input_file, 'r', encoding='utf-8') as fin, open(dpath + output_file, 'w', encoding='utf-8') as fout:
#         unique_conv_id = 0

#         for line in tqdm(fin):
#             conversation = json.loads(line)
#             dialogs = conversation["dialog"]
#             movieid2name = conversation["movieMentions"]
#             contexts = []
#             entities = []

#             for i, message in enumerate(dialogs):
#                 entity_ids = [entity2id[entity] for entity in message["entity"] if entity in entity2id]
#                 movie_ids = [entity2id[movie] for movie in message["movies"] if movie in entity2id]
                
#                 entities.extend(entity_ids)
#                 entities.extend(movie_ids)
                
#                 if message["role"] == "Seeker":
#                     # @movie_id => movie_name
#                     utt = process_utt(message["text"], movieid2name, replace_movieId=True)
#                     contexts.append(utt)  # Seeker의 발화만 context에 추가
                
#                 if message["role"] == "Seeker":
#                     # # 다음 message가 대화의 마지막 message이면서 "role"이 "Recommender" 일 때
#                     # if i + 1 == len(dialogs) - 1 and dialogs[i + 1]["role"] == "Recommender":
#                     #     next_entities = [entity2id[entity] for entity in dialogs[i + 1]["entity"] if entity in entity2id]
#                     #     entities.extend(next_entities)
                        
#                     # Write the current context and entities
#                     fout.write(json.dumps({
#                         "conv_id": conversation["conv_id"],
#                         #"unique_conv_id": str(unique_conv_id),
#                         "contexts": contexts,
#                         "entities": list(set(entities))
#                     }, ensure_ascii=False) + '\n')
                    
#                     # Prepare for the next context
#                     unique_conv_id += 1
#                     contexts = []
#                     entities = [] # Reset entities for the new Seeker

########################################################################

if __name__ == '__main__':
    import sys
    sys.path.append('/home/hyuns6100/[4]newCRS/')
    
    from config import parse_args
    args = parse_args()
    
    dpath = args.dataset
    
    with open(dpath + 'entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)

    reformat_data('senti_train_data_processed.jsonl', 'only_user_uni_item_reformat_senti_train_data_processed.jsonl', dpath)
    reformat_data('senti_valid_data_processed.jsonl', 'only_user_uni_item_reformat_senti_valid_data_processed.jsonl', dpath)
    reformat_data('senti_test_data_processed.jsonl', 'only_user_uni_item_reformat_senti_test_data_processed.jsonl', dpath)

    print("> > > > > End of Processing for sentiment classification < < < < <", end='\n')