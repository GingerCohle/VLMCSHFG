# VLMCSHFG
This is the official code repo for "Vision-Language Models Empowered Nighttime Object Detection  with Consistency Sampler and Hallucination Feature Generator"
# Vision-Language Models Empowered Nighttime Object Detection  with Consistency Sampler and Hallucination Feature Generator (Under Review)



## Installation

####  Our work is based on Python 3.7 and Pytorch 1.7.1+cu111 due to the  [CLIP requirement](https://github.com/openai/CLIP). The hardware is Nvidia Tesla V100 single GPU. Give a big thanks to Dr. Li WuYang with his work [SIGMA](https://github.com/CityU-AIM-Group/SIGMA). We use it as baseline. The trained pth are  uploaded to [google drive](https://drive.google.com/drive/folders/1pMiPDe1If7rssy6332jL6lu_MAzy8UXo?usp=drive_link).
![image](https://github.com/user-attachments/assets/9d9c40fc-6f68-4742-a9e1-3af3512f473c)

#### Basic Installation

```bash
conda create -n vldadaptor  python==3.7 -y
conda activate vldadaptor
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
git clone https://github.com/GingerCohle/VLMCSHFG.git
cd VLMCSHFG
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
conda install ipython
pip install pynndescent
conda install scipy==1.3.1 -y
pip install ninja yacs cython matplotlib tqdm 
pip install --no-deps torchvision==0.2.1 
python setup.py build_ext install
cd ../..
pip install opencv-python==3.4.17.63
pip install scikit-learn
pip install scikit-image
python setup.py build develop
pip install Pillow==7.1.0
pip install tensorflow tensorboardX
pip install ipdb
```

#### CLIP Installation (China Region)

```bash
pip install ftfy regex tqdm
pip install git+https://gitee.com/lazybeauty/CLIP.git
```

#### CLIP Installation (Other Regions)

```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Setting

#### Before Training, you should prepare for the dataset, [SHIFT, BDD100k](https://www.dropbox.com/scl/fo/258uzp6i0dz17zsj234r6/h?rlkey=kb6brfk1oqc1ddsa3ulz8v9ei&e=1&dl=0) and FLIR. Thanks to the authors of [2PCNET](https://github.com/mecarill/2pcnet). The FLIR dataset has been uploaded to [Baidu Cloud]( https://pan.baidu.com/s/1_sjia3kP-JGs69W0AxzYoQ) with password c555.

#### After dataset preparation, you can train the model by change the yaml [config file](https://github.com/GingerCohle/VLMCSHFG/blob/main/configs/VLMCSHFG/vlmcshfg_res50_cityscapace_to_foggy.yaml#L53) for class number and [dataset path file](https://github.com/GingerCohle/VLMCSHFG/blob/main/fcos_core/config/paths_catalog.py#L99).

## Training

#### SHIFT and BDD100k Training (Cuda Device 0 with Single Gpu)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net_da.py --config-file configs/VLMCSHFG/vlmcshfg_res50_cityscapace_to_foggy.yaml
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file configs/VLMCSHFG/vlmcshfg_res50_cityscapace_to_foggy.yaml MODEL.WEIGHT $model path$
```

## Code Inside

#### CCCS and VLMOE.
![image](https://github.com/user-attachments/assets/bdf611ee-637b-40d0-92d9-dd4bd3a26f19)



#### CCCS clusting in  [loss](https://github.com/GingerCohle/VLMCSHFG/blob/main/fcos_core/modeling/rpn/fcos/loss.py) 


```python
def clust_rank(mat, initial_rank=None, distance='cosine'):
    s = mat.shape[0]
    mat = mat.cpu().detach().numpy()
    if initial_rank is not None:
        orig_dist = []
    elif s <= ANN_THRESHOLD:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')

        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust
```

#### VLMOE in  [graph_matching_head](https://github.com/GingerCohle/VLMCSHFG/blob/main/fcos_core/modeling/rpn/fcos/graph_matching_head.py) 

##### OPL loss

```python
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
       super(OrthogonalProjectionLoss, self).__init__()
       self.gamma = gamma

    def forward(self, features, labels=None):
       device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

       #  features are normalized
       features = F.normalize(features, p=2, dim=1)

       labels = labels[:, None]  # extend dim

       mask = torch.eq(labels, labels.t()).bool().to(device)
       eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

       mask_pos = mask.masked_fill(eye, 0).float()
       mask_neg = (~mask).float()
       dot_prod = torch.matmul(features, features.t())

       pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
       neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

       loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

       return loss
```

##### The VLMOE are combined with CLIP in  [graph_matching_head](https://github.com/GingerCohle/VLMCSHFG/blob/main/fcos_core/modeling/rpn/fcos/graph_matching_head.py) 




#### Hallucination Feature Generator
![image](https://github.com/user-attachments/assets/2b4d6f3d-a748-4125-bb8f-ade0dd1285fc)

 all defined in [graph_matching_head](https://github.com/GingerCohle/VLMCSHFG/blob/main/fcos_core/modeling/rpn/fcos/graph_matching_head.py) 

```python
for cls in tg_labels.unique().long():
    origin_node = tg_nodes[tg_labels == cls].detach()
    nodes_nums = len(origin_node)
    if nodes_nums<10:
        #print("tg",nodes_nums,"pass")
        continue
    generate_node = self.tg_seed[cls].unsqueeze(0).expand(nodes_nums, 256)
    #seed
    rng_state = torch.get_rng_state()  # global seed
    temp_seed = self.tg_seed[cls][0]  
    torch.manual_seed(temp_seed)
    noise = torch.randn(nodes_nums, 256).cuda()  # noise gen
    generate_node = noise + generate_node
    torch.set_rng_state(rng_state)  # global seed
    #
    generate_node = self.SeNode_generator(generate_node)

    # kl cal
    prob1 = F.softmax(origin_node, dim=1)
    prob2 = F.softmax(generate_node, dim=1)
    kl_loss = F.kl_div(prob1.log(), prob2, reduction='batchmean')
    #print("KL Divergence:", kl_div)
    kl_all.append(kl_loss.unsqueeze(0))


#print("generate_loss", kl_all)
generate_loss = torch.cat(kl_all,dim=0)
generate_loss = torch.mean(generate_loss)
#print("generate_loss",generate_loss)

return generate_loss
```

