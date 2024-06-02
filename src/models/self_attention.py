import torch
import torch.nn as nn

class Self_Attention_3D(nn.Module):
    """ Self-AttentionのLayer for 3D data """

    def __init__(self, in_dim):
        super(Self_Attention_3D, self).__init__()

        # 1×1×1の畳み込み層によるpointwise convolutionを用意
        self.query_conv = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Attention Map作成時の規格化のソフトマックス
        self.softmax = nn.Softmax(dim=-2)

        # 元の入力xとSelf-Attention Mapであるoを足し算するときの係数
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 入力変数
        X = x

        # 畳み込みをしてから、サイズを変形する。 B,C',D,W,H → B,C',N へ
        proj_query = self.query_conv(X).view(
            X.shape[0], -1, X.shape[2]*X.shape[3]*X.shape[4])  # サイズ：B,C',N
        proj_query = proj_query.permute(0, 2, 1)  # 転置操作
        proj_key = self.key_conv(X).view(
            X.shape[0], -1, X.shape[2]*X.shape[3]*X.shape[4])  # サイズ：B,C',N

        # かけ算
        S = torch.bmm(proj_query, proj_key)  # bmmはバッチごとの行列かけ算です

        # 規格化
        attention_map_T = self.softmax(S)  # 行i方向の和を1にするソフトマックス関数
        attention_map = attention_map_T.permute(0, 2, 1)  # 転置をとる

        # Self-Attention Mapを計算する
        proj_value = self.value_conv(X).view(
            X.shape[0], -1, X.shape[2]*X.shape[3]*X.shape[4])  # サイズ：B,C,N
        o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))

        # Self-Attention MapであるoのテンソルサイズをXにそろえて、出力にする
        o = o.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        out = x + self.gamma * o

        return out, attention_map
