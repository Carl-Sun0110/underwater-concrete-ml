"""可被sklearn分组CV克隆的CPU小型神经网络，显式实现Dropout。"""
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.utils.validation import check_X_y,check_array,check_is_fitted


class TorchDropoutRegressor(RegressorMixin,BaseEstimator):
    """固定轮数训练；不接收外层验证/测试目标，目标缩放交给折内TTR。"""
    def __init__(self,hidden_sizes=(32,16),dropout=.1,learning_rate=.003,weight_decay=.001,
                 epochs=200,batch_size=64,random_state=42,torch_threads=1):
        self.hidden_sizes=hidden_sizes;self.dropout=dropout;self.learning_rate=learning_rate
        self.weight_decay=weight_decay;self.epochs=epochs;self.batch_size=batch_size
        self.random_state=random_state;self.torch_threads=torch_threads

    def fit(self,X,y):
        import torch
        from torch import nn
        if not 0<=self.dropout<1 or self.epochs<1 or self.batch_size<1 or self.torch_threads<1:
            raise ValueError('网络训练参数无效')
        if not self.hidden_sizes or any(n<1 for n in self.hidden_sizes):raise ValueError('隐层结构无效')
        X,y=check_X_y(X,y,dtype=np.float32,y_numeric=True)
        self.n_features_in_=X.shape[1]
        torch.set_num_threads(self.torch_threads)
        torch.manual_seed(self.random_state)
        torch.use_deterministic_algorithms(True)
        layers=[];previous=self.n_features_in_
        for width in self.hidden_sizes:
            layers.extend([nn.Linear(previous,width),nn.ReLU(),nn.Dropout(self.dropout)])
            previous=width
        layers.append(nn.Linear(previous,1));self.network_=nn.Sequential(*layers).cpu()
        features=torch.tensor(X,dtype=torch.float32);targets=torch.tensor(y,dtype=torch.float32).reshape(-1,1)
        optimizer=torch.optim.AdamW(self.network_.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)
        loss_fn=nn.MSELoss();generator=torch.Generator().manual_seed(self.random_state)
        self.loss_curve_=[];self.network_.train()
        for epoch in range(self.epochs):
            permutation=torch.randperm(len(X),generator=generator);total=0.
            for start in range(0,len(X),self.batch_size):
                indices=permutation[start:start+self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(self.network_(features[indices]),targets[indices])
                if not torch.isfinite(loss):raise ValueError('神经网络损失非有限值')
                loss.backward();optimizer.step();total+=loss.item()*len(indices)
            self.loss_curve_.append(total/len(X))
        self.n_iter_=self.epochs;self.network_.eval()
        return self

    def predict(self,X):
        import torch
        check_is_fitted(self,'network_');X=check_array(X,dtype=np.float32)
        if X.shape[1]!=self.n_features_in_:raise ValueError('输入列数不匹配')
        self.network_.eval()
        with torch.inference_mode():pred=self.network_(torch.tensor(X,dtype=torch.float32)).cpu().numpy().ravel()
        if not np.isfinite(pred).all():raise ValueError('预测非有限值')
        return pred
