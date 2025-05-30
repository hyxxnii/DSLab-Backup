import json
import os
import re
import html
from collections import defaultdict
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


def asc_process_data(input_file, output_file, dpath):
    
    with open(dpath + input_file, 'r', encoding='utf-8') as fin, open(dpath + output_file, 'w', encoding='utf-8') as fout:
        unique_conv_id = 0

        for line in tqdm(fin):
            conversation = json.loads(line)
            contexts = []
            entities = defaultdict(dict) 
            aspects = defaultdict(dict)
            polarity = defaultdict(dict)
            dialogs = conversation["dialog"]
            movieid2name = conversation["movieMentions"]
            seekerPolarity = conversation["seekerPolarity"]

            for i, message in enumerate(dialogs):
                # @movie_id => movie_name
                utt = process_utt(message["text"], movieid2name, replace_movieId=True)
                
                role_prefix = "User: " if message["role"] == "Seeker" else "System: "
                formatted_text = role_prefix + utt

                # Update context and entities
                contexts.append(formatted_text)
                # contexts.extend(["</s>"])
                
                for i, entity in enumerate(message["entities"]):
                    if entity in entity2id:
                        entity_id = entity2id[entity]
                        entities[entity_id] = message["entity_names"][i]
                
                # if len(message["movies"]) != 0:
                for movie in message["movies"]:
                    if movie in entity2id:
                        entity_mv_id = entity2id[movie]
                        # redial_mv_id = new_entityid_redial_movie_id[str(entity_mv_id)]
                        try:
                            redial_mv_id = new_entityid_redial_movie_id[str(entity_mv_id)]
                            movie_title = movieid2name[redial_mv_id]
                        except:
                            # # redial_mv_id = re.findall(r'@(\d+)', message["text"])
                            # print(conversation["conv_id"], message["utt_id"], "entityid:", entity_mv_id, "redialid:", redial_mv_id)
                            for k, v in new_redial_movie_id2_entityid.items():
                                if v == entity_mv_id:
                                    try: # 동일 entity id가 여러 key (original redial movie id)에 매핑되었을 수도 있음
                                        movie_title = movieid2name[k]
                                        redial_mv_id = k
                                    except:
                                        continue
                                    
                        aspects[entity_mv_id] = movie_title
                        polarity[entity_mv_id] = seekerPolarity[redial_mv_id]['liked']
                
                if message["role"] == "Seeker" and len(aspects) != 0:
                    try:
                        assert len(set(aspects)) == len(set(polarity))
                    except:
                        print("길이 불일치: ", conversation["conv_id"], message["utt_id"])
                    
                    # Write the current context and entities
                    fout.write(json.dumps({
                        "conv_id": conversation["conv_id"],
                        "uni_conv_id": str(unique_conv_id),
                        "contexts": contexts,
                        "aspects": aspects,
                        "polarity": polarity
                    }, ensure_ascii=False) + '\n')
                    
                    # Prepare for the next context
                    unique_conv_id += 1
                    contexts = [] # 누적 X
                    aspects = defaultdict(dict) # Reset entities for the new Seeker
                    polarity = defaultdict(dict)
                    
                    
                    
if __name__ == '__main__':
    import sys
    sys.path.append('/home/hyuns6100/[4]newCRS')
    from config import parse_args
    args = parse_args()
    
    dpath = "/home/hyuns6100/[4]newCRS/data/redial/"

    entity2id = json.load(
                open(os.path.join(dpath, 'entity2id.json'), 'r', encoding='utf-8'))

    new_redial_movie_id2_entityid = json.load(
                open(os.path.join(dpath, 'new_redial_movie_id2_entityid.jsonl'), 'r', encoding='utf-8'))

    new_entityid_redial_movie_id = json.load(
                open(os.path.join(dpath, 'new_entityid_redial_movie_id.jsonl'), 'r', encoding='utf-8'))


    asc_process_data('senti_train_data_processed.jsonl', 'tmp_asc_train_data_processed.jsonl', dpath)
    asc_process_data('senti_valid_data_processed.jsonl', 'tmp_asc_valid_data_processed.jsonl', dpath)
    asc_process_data('senti_test_data_processed.jsonl', 'tmp_asc_test_data_processed.jsonl', dpath)