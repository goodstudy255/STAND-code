#coding=utf-8

class Sample(object):
    '''
    一个样本。
    '''
    def __init__(self):
        self.id = -1           #第几个窗口
        self.session_id = -1   #user_id的值
        # self.video_id = -1     # video_id的值
        self.click_items = []    # 窗口内video_id的list
        self.in_play_time_ms = []
        self.in_duration_ms = []

        self.in_label = []
        self.out_label = []
        
        self.out_play_time_ms = []
        self.out_duration_ms = []

        self.items_idxes = []

        self.in_tag = []
        self.out_tag = []

        self.in_idxes = []
        self.out_idxes = []
        self.label = []
        self.pred =[]
        self.best_pred = []
        self.is_click_in_list = []
        self.is_click_out_list = []
        self.ext_matrix = {} # 额外数据，key是名字，value是矩阵。例如attention.   'alpha':[]

    def __str__(self):
        ret = 'id: ' + str(self.id) + '\n'
        ret += 'session_id: ' + str(self.session_id) + '\n'
        ret += 'items: '+ str(self.items_idxes) + '\n'
        ret += 'click_items: '+ str(self.click_items) + '\n'
        ret += 'in_play_time_ms_list:'+ str(self.in_play_time_ms)+'\n'
        ret += 'in_duration_ms_list:'+ str(self.in_duration_ms)+'\n'
        ret += 'out_play_time_ms_list:'+ str(self.out_play_time_ms)+'\n'
        ret += 'out_duration_ms_list:'+ str(self.out_duration_ms)+'\n'
        ret += 'out: ' + str(self.out_idxes) + '\n'
        ret += 'in: '+ str(self.in_idxes) + '\n'
        ret += 'label: '+ str(self.label) + '\n'
        return ret