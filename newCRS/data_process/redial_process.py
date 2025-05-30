from tqdm.auto import tqdm
import json

def extract_ids(items, entity2id):
    return [entity2id[item] for item in items if item in entity2id]

def process(input_file, output_file, dpath, movie_set):
    with open(dpath + input_file, 'r', encoding='utf-8') as fin, open(dpath + output_file, 'w', encoding='utf-8') as fout:
        # unique_conv_id = 0

        for line in tqdm(fin):
            conversation = json.loads(line)
            dialogs = conversation["dialog"]
            contexts = []
            resp = ''
            entities = []

            for i, message in enumerate(dialogs):
                entity_ids = extract_ids(message["entity"], entity2id)
                movie_ids = extract_ids(message["movies"], entity2id)
                movie_set |= set(movie_ids)
                
                if message['role'] == 'Seeker':
                    contexts.append(message['text'])                    
                    entities.extend(entity_ids)
                    entities.extend(movie_ids)
                    
                elif message['role'] == 'Recommender':
                    # 첫 recommender 발화인 경우 무시
                    if i == 0: 
                        contexts.append(message['text'])
                        entities.extend(entity_ids + movie_ids)
                        continue
                    
                    resp = message["text"]

                    fout.write(json.dumps({
                        "conv_id": conversation["conv_id"],
                        "contexts": contexts.copy(),
                        "resp": resp, # recommender의 발화
                        "entities": list(set(entities.copy())), # 이전 발화에서 언급된 entity, movie 리스트
                        "rec": list(set(movie_ids)) # recommender의 발화에 언급된 movie (ground-truth) 리스트 
                    }, ensure_ascii=False) + '\n')

                    # 현재 recommender 발화를 contexts에 누적
                    contexts.append(resp) # message['text'])
                    entities.extend(list(set(entity_ids + movie_ids)))
                    
                    # 마지막 utterance가 Seeker의 발화인 경우 무시
                    if i == len(conversation['dialog']) - 2 and conversation['dialog'][i + 1]['role'] == 'Seeker':
                        continue



if __name__ == '__main__':
    import sys
    sys.path.append('/home/hyuns6100/[4]newCRS')
    from config import parse_args
    args = parse_args()
    
    with open(args.dataset + 'entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)
    
    movie_set = set()
    # with open('node2text_link_clean.json', 'r', encoding='utf-8') as f:
    #     node2entity = json.load(f)


    ## 추천 모델 (언급된 entity,item) 과 생성 모델 (이전 발화, 추천 발화(ground-truth)) 를 위한 전처리


    process('senti_train_data_processed.jsonl', 'train_data_processed.jsonl', args.dataset, movie_set)
    process('senti_valid_data_processed.jsonl', 'valid_data_processed.jsonl', args.dataset, movie_set)
    process('senti_test_data_processed.jsonl', 'test_data_processed.jsonl', args.dataset, movie_set)

    with open(args.dataset + 'item_ids.json', 'w', encoding='utf-8') as f:
        json.dump(list(movie_set), f, ensure_ascii=False)
    
    print(f'#movie item: {len(movie_set)}')
