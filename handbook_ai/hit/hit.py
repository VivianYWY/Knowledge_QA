import pandas as pd
from utils.similarity import jaccard_similarity
import config.params as conf

class HitUse:
    def __init__(self):
        self.question_list = None

    def question_list_check(self, query):
        is_hit = False
        preset_answer = ''
        for i in range(0,len(self.question_list)):
            if jaccard_similarity(query, self.question_list.iloc[i]['question']) > conf.hit_simi:
                is_hit = True
                preset_answer = self.question_list.iloc[i]['answer']
                break

        return is_hit, preset_answer

    def load_question_list(self, filepath):
        file_format = filepath.split('.')[-1]
        if file_format in ['xls', 'xlsx']:  # Process excel file
            self.question_list = pd.read_excel(filepath)
        else:  # Do not support other format
            raise Exception("目前暂不支持" + file_format)

