import torch
from torch import nn

class CNNLSTMModel(nn.Module):

    def __init__(self, lstm_hidden_size=256, lstm_layers=2, dropout=0.1, max_sequence_length=50):
        super(CNNLSTMModel, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.max_sequence_length = max_sequence_length
        

        self.cnn = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 全局池化
            nn.AdaptiveAvgPool2d((4, 9)),
            nn.Flatten()
        )
        
   
        self.cnn_output_size = 64 * 4 * 9 #4种花色,9张牌
        

        self.feature_compressor = nn.Sequential(
            nn.Linear(self.cnn_output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False 
        )
        
        # 注意力机制（关注重要的历史时刻）
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 残差
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_hidden_size + 256, 512),  # 拼接当前特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 235)
        )
        

        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_size + 256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input_dict, hidden_state=None, return_attention=False):
        """
        Args:
            input_dict: 包含observation和action_mask
            hidden_state: LSTM隐藏状态 (h_0, c_0)
            return_attention: 是否返回注意力权重
        """
        obs = input_dict['observation'].float()
        

        if len(obs.shape) == 4:  
            obs = obs.unsqueeze(1)  
        
        batch_size, seq_len, channels, height, width = obs.size()
        

        cnn_input = obs.view(batch_size * seq_len, channels, height, width)
        cnn_features = self.cnn(cnn_input)  
        

        compressed_features = self.feature_compressor(cnn_features) 
        compressed_features = compressed_features.view(batch_size, seq_len, 256)
        

        if hidden_state is None:
            lstm_output, new_hidden = self.lstm(compressed_features)
        else:
            lstm_output, new_hidden = self.lstm(compressed_features, hidden_state)
        

        attended_output, attention_weights = self.attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # 取最后一个时间步
        lstm_last = attended_output[:, -1, :]  
        current_features = compressed_features[:, -1, :]  
        
        # 拼接LSTM输出和当前特征
        combined_features = torch.cat([lstm_last, current_features], dim=1)
        

        logits = self.policy_head(combined_features)
        

        action_mask = input_dict["action_mask"]
        if len(action_mask.shape) > 2:
            action_mask = action_mask[:, -1, :]  
        
        mask = action_mask.float()
        inf_mask = torch.clamp(torch.log(mask + 1e-8), -1e38, 1e38)
        masked_logits = logits + inf_mask
        

        value = self.value_head(combined_features)
        
        if return_attention:
            return masked_logits, value, new_hidden, attention_weights
        else:
            return masked_logits, value, new_hidden