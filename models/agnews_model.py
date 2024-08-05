import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel,BertModel
    
class ClientNet(nn.Module):
    def __init__(self):
        super(ClientNet, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.embeddings = bert.embeddings

    def forward(self, input):
        input_ids = input[:,0,:]
        attention_mask = input[:,1,:]
        embeddings = self.embeddings(input_ids=input_ids.long())
        output = torch.cat([attention_mask,torch.flatten(embeddings, 1)],dim=1)
        return output
    
class ServerNet(nn.Module):
    def __init__(self,party_num,mask_dim,output_dim):
        super(ServerNet, self).__init__()
        self.party_num = party_num
        self.mask_dim = mask_dim
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.config = bert.config
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.get_head_mask = bert.get_head_mask
        self.get_extended_attention_mask = bert.get_extended_attention_mask
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, output_dim)

    def forward(self,input):
        l = int(input.shape[1]/self.party_num)
        attention_mask = []
        embeddings = []
        for i in range(0,self.party_num):
            data = input[:,i*l:(i+1)*l]
            attention_mask.append(data[:,:self.mask_dim])
            embeddings.append(data[:,self.mask_dim:].view(input.shape[0],-1,768))
        attention_mask = torch.cat(attention_mask,dim=1)
        embeddings = torch.cat(embeddings,dim=1)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, attention_mask.size())
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(embeddings,attention_mask=extended_attention_mask,head_mask=head_mask,)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        last_hidden_state,pooled_output = (sequence_output, pooled_output) + encoder_outputs[1:]
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_output = F.relu(linear_output)
        return final_output