---
title: 台大李宏毅-深度学习课程笔记-2：Improving GAN
categories:
- 台大李宏毅-深度学习
- 生成模型
tags:
- GAN
updated: 2018-11-27
---
## 1 GAN Framework
### 1.1 f-divergence
2017年提出的[f-GAN](https://arxiv.org/abs/)可以总结为一句话：衡量两个分布（$P_data,P_G$）差异，除了 $JS-divergence$ 外，你可以选择任何 $f-divergence$。但是，什么是 $f-divergence$ 呢？  
假如有两个分布 $P$ 和 $Q$，$f-divergence$ 就是衡量两个分布有多不一样，$f-divergence$ 的定义如下：  
$$D_f(P||Q)=\int_xq(x)f\left(\frac{p(x)}{q(x)}\right)dx$$
其中，$f$ 函数满足两点：  

- f是凸函数
- $f(1)=0$  

![](/assets/blog_images/2018-11-27/f-divergence1.png)  
上图根据凸函数的性质可以证明 $f-divergence\geqslant0$，当$p(x)=q(x)$时，$f-divergence=0$  
实际上，$KL\ divergence$也是一个$f-divergence$  
![](/assets/blog_images/2018-11-27/f-divergence2.png)

### 1.2 Fenchel Conjugate  
每一个凸函数 $f$ 都有一个 $Conjugate\ Function\ f^\ast$:  
$$f^\ast(t)=\mathop{max}_{x\in dom(f)}\lbrace xt-f(x)\rbrace$$  
通过假设不同的 $t$ 代入 $f^\ast$，然后求导得到最大值，$f^\ast$ 的样子如下图：
![](/assets/blog_images/2018-11-27/FC1.png)  
当$f(x)=x\log(x)$时，$f^\ast(t)=\exp^{t-1}$:
![](/assets/blog_images/2018-11-27/FC2.png)  
**特别强调：$f(f^\ast)^\ast=f$**
![](/assets/blog_images/2018-11-27/FC3.png)  

### 1.3 Connect to GAN
因为 $f-divergence$ 中的 $f$ 是凸函数，可以将它转换成它的 $Conjugate\ Function\ f^\ast$
$$
\begin{split}
D_f(P||Q)&=\int_xq(x)f\left(\frac{p(x)}{q(x)}\right)dx\\
         &=\int_xq(x)\left(\mathop{max}_{t\in dom (f^\ast)}\left\lbrace \frac{p(x)}{q(x)}t-f^\ast(t)\right\rbrace\right)dx
\end{split}
$$  
假设有一个函数$D(x)$，它的输入为$x$，输出为$t$，根据凸函数的性质：

$$
\begin{split}
D_f(P||Q)&\geqslant\mathop{max}_{D}\int_xq(x)\left(\frac{p(x)}{q(x)}D(x)-f^\ast(D(x)) \right)dx\\
        &=\mathop{max}_{D}\int_xp(x)D(x)dx-\int_xq(x)f^\ast(D(x))dx\\
        &=\mathop{max}_{D}\lbrace E_{x\sim p(x)}[D(x)]-E_{x\sim q(x)}[f^\ast(D(x))]\rbrace
\end{split}
$$   

回到 GAN 问题上，将 $p(x)$ 和 $q(x)$ 换成 $P_{data}$ 和 $P_G$：  
$$D_f(P_{data}||P_G)=\mathop{max}_{D}\lbrace E_{x\sim P_{data}(x)}[D(x)]-E_{x\sim P_G(x)}[f^\ast(D(x))]\rbrace$$  

给定一个 $P_{data}$，找一个 $P_G$ 使得它们尽可能地相近，则：
$$
\begin{split}
G^\ast&=arg\,\mathop{min}_GD_f(P_{data},P_G)\\
      &=arg\,\mathop{min}_G\,\mathop{max}_{D}\lbrace E_{x\sim P_{data}(x)}[D(x)]-E_{x\sim P_G(x)}[f^\ast(D(x))]\rbrace\\
      &=arg\,\mathop{min}_G\,\mathop{max}_DV(G,D)
\end{split}
$$  

从上面可以看出，原来的GAN只是 $f-divergence$ 的一个特例，我们想要Minimize哪个 $divergence$，只需找出它的 $Conjugate\ Function\ f^\ast$ 替换进去就可以了，这样一个完整的GAN Framework就产生了。文中尝试了不同的$divergence$。  
![](/assets/blog_images/2018-11-27/CG2.png)  
比如GAN：
$$
\begin{split}
&\int p(x)\log\frac{2p(x)}{p(x)+q(x)}+q(x)\log\frac{2q(x)}{p(x)+q(x)}dx-\log(4)\\
=&\int q(x)\frac{p(x)}{q(x)}\log\frac{p(x)}{p(x)+q(x)}+q(x)\log\frac{q(x)}{p(x)+q(x)}dx\\
=&\int -q(x)\frac{p(x)}{q(x)}\log\frac{p(x)+q(x)}{p(x)}-q(x)\log\frac{p(x)+q(x)}{q(x)}dx\\
=&\int q(x)\left(-\frac{p(x)}{q(x)}\log\left(\frac{1+\frac{p(x)}{q(x)}}{\frac{p(x)}{q(x)}}\right)-log\left(1+\frac{p(x)}{q(x)}\right)\right)dx\\
=&\int q(x)\left(\frac{p(x)}{q(x)}\log\frac{p(x)}{q(x)}-\frac{p(x)}{q(x)}\log\left(1+\frac{p(x)}{q(x)}\right)-\log\left(1+\frac{p(x)}{q(x)}\right)\right)dx\\
=&\int q(x)\left(\frac{p(x)}{q(x)}\log\frac{p(x)}{q(x)}-\left(1+\frac{p(x)}{q(x)}\right)\log\left(1+\frac{p(x)}{q(x)}\right)\right)dx
\end{split}
$$  
因此，GAN的$f$函数和$f^\ast$函数为:  
$$
\begin{split}
&f=u\log u-(1+u)\log(1+u)\\
&f^\ast=-log(1-exp(t))
\end{split}
$$

实验结果表明选择不同的$divergence$模型拟合的效果也不一样：  
![](/assets/blog_images/2018-11-27/CG3.png)  
f-GAN 还做了一处改进，原来的 GAN 求解 $D^\ast$ 需要重复 k 次update，求解 $G^\ast$ 只需一次update，而作者证明了求解 $D^\ast$ 和 $G^\ast$ 时都进行一次update也是可以收敛的。  
![](/assets/blog_images/2018-11-27/CG1.png) 

## 2 WGAN
一句话说明[WGAN](https://arxiv.org/abs/1701.07875)：传统GAN的目的是Minimize两个 distribution 的 $f-divergence$，而 WGAN 是Minimize两个 distribution 的 $Earth\ Mover’s\ Distance$。而 $Earth\ Mover’s\ Distance$ 其实就是 $Wasserstein\ Distance$，这也是WGAN名字的由来。 
### 2.1 Original WGAN  
$Earth\ Mover’s\ Distance$是指将一个分布 $P$ 通过搬运的方式变成另一个分布 $Q$ 所需要的最少搬运代价。如下图一维空间上两个分布的距离是 $d$，那么 $W(P,Q)=d$。  
![](/assets/blog_images/2018-11-27/EMD1.png)   
如果两个分布比较复杂，比如下图的分布 $P$ 和 $Q$，有很多种方法将分布 $P$ 铲成分布 $Q$，而每一种方法称之为“moving plan”，不同方法的EM距离是不一样，下图中左边方法的EM距离明显小于右边的EM距离，因为右边有点舍近求远。而我们的目的是寻找平均距离最小的“moving plan”。
![](/assets/blog_images/2018-11-27/EMD2.png)   
一个“moving plan”可以用一个矩阵$\lambda$表示，矩阵中的元素$\lambda_{ij}$表示将 $P$ 铲成 $Q$ 的时候从 $P$ 中第i个位置移动到 $Q$ 中第j个位置移动的总量。矩阵中的每一行i的总和等于$P_i$，每一列j的总和等于$Q_j$。给定一个“moving plan”，那么它的平均距离的定义为：  
$$B(\lambda)=\sum_{x_p,x_q}\lambda(x_p,x_q)||x_p-x_q||$$  

穷举所有的“moving plan”，平均距离最小的那个值就是$W(P,Q)$。  
![](/assets/blog_images/2018-11-27/EMD3.png)  
为什么使用EM距离呢？因为它没有 $JS\ Divergence$ 的问题。比如说，当第 0、50、100 次迭代时，两个分布的变化是这样：  
![](/assets/blog_images/2018-11-27/EMD4.png)  
上图训练期间能看出来迭代过程中 JSD 总是不变的（永远是log2），直到两个分布重叠的一瞬间，JSD 降为0。而 EM 距离即便在两次迭代中两个分布完全没有重叠，但一定有 EM 距离上的区别。 
在[令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)中对EM距离的数学定义有比较通俗的解析：  
![](/assets/blog_images/2018-11-27/EMD5.png)   
既然 EM 距离有如此优越的性质，如果我们能够把它定义为生成器的 $loss$，就可以产生有意义的梯度来更新生成器，使得生成分布被拉向真实分布。  
WGAN作者用了一个已有的定理把 EM 距离变换为如下形式：  
$$W(P_{data},P_G)=\frac{1}{K}\mathop{sup}_{||f||_L\leq K}E_{x\sim P_{data}}[f(x)]-E_{x\sim P_G}[f(x)]$$  
将 EM 距离与 GAN 联系起来：  
![](/assets/blog_images/2018-11-27/BGF1.png)  
上图中可以发现，原来的 GAN 里 $D$ 是没有任何限制的，而 WGAN 中的 $D$ 要符合 $1-Lipschitz$，$1-Lipschitz$ 表示一个函数集，当 $f$ 是一个 $Lipschitz$ 函数时，它应该受到以下约束：  
$$||f(x_1)-f(x_2)||\leq K||x_1-x_2||$$  
而 $1−Lipschitz$ 函数中的 $K=1$。直观来说，就是让这个函数的变化“缓慢一些”。  
为什么要对生成器 $D$ 做限制呢？假设我们现在有两个一维的分布，$x_1$ 和 $x_2$ 的距离是 $d$，显然他们之间的 EM 距离也是 $d$：
![](/assets/blog_images/2018-11-27/BGF2.png)  
我们要去优化$W(P_{data},P_G)$：  
$$W(P_{data},P_G)=\mathop{max}_{D\in 1-Lipschitz}\lbrace E_{x\sim P_{data}}[f(x)]-E_{x\sim P_G}[f(x)]\rbrace$$  

只需让 $D(x_1)=+\infty$，而 $D(x_2)=-\infty$ 即可。但这样子会导致模型训练非常困难：判别器区分能力太强，很难驱使生成器提高生成分布数据质量。  
如果加上了 $Lipschitz$ 限制，$||D(x_1)-D(x_2)||\leq||x_1-x_2||=d$，那么判别器在不同分布上的结果限制在了一个较小的范围中。传统的 GAN 所使用的判别器是一个最终经过 $sigmoid$ 输出的神经网络，它的输出曲线肯定是一个 S 型。在真实分布附近是 1，在生成分布附近是 0。而现在我们对判别器施加了这个限制，同时不用在最后一层使用 $sigmoid$，它有可能是任何形状的线段，只要能让 $||D(x_1)-D(x_2)||\leq d$ 即可。如下图所示：  
![](/assets/blog_images/2018-11-27/BGF3.png)
传统 GAN 的判别器是有饱和区的（靠近真实分布和生成分布的地方，函数变化平缓，梯度趋于 0）。而现在的 WGAN 如果是一条直线，那就能在训练过程中无差别地提供一个有意义的梯度。  
如何实现判别器 $D$ 的 $Lipschitz$ 限制呢？作者采取了一个非常简单的做法，就是限制神经网络 $f_\theta$ 的所有参数 $w_i$ 不超过某个范围 $[-c,c]$。但这么做实际上保证的并不是 $1-Lipschitz$ 而是 $K-Lipschitz$。  
**WGAN的算法流程如下：**  
![](/assets/blog_images/2018-11-24/2018-11-27-BGF4.png)  
WGAN的 $Discriminator\ loss$ 可以衡量模型的优劣，如下图中，$loss$ 越小，生成的图像越清晰。  
![](/assets/blog_images/2018-11-27/BGF4.png)
### 2.2 Improving WGAN
在最初的 WGAN 中，通过截断权重的方法来实现 对判别器 $D$ 实现 $1−Lipschitz$ 的等效限制。$1−Lipschitz$ 函数有一个特性：当一个函数是 $1−Lipschitz$ 函数时，它的梯度的 norm 将永远小于等于 1。  
$$D\in 1−Lipschitz\Longleftrightarrow||\nabla_xD(x)||\leq1\ for\ all\ x$$ 这个特性可以将原来的WGAN转换成下面的样子：
$$
\begin{split}
W(P_{data},P_G)=&\mathop{max}_{D}\lbrace E_{x\sim P_{data}}[f(x)]-E_{x\sim P_G}[f(x)]\\
&-\lambda\int_xmax(0,||\nabla_xD(x)||-1)\rbrace
\end{split}
$$   
现在我们寻找判别器 $D$ 的函数集不再是$1−Lipschitz$中的函数了，而是任意函数，只是后面增加了一项惩罚项。这个惩罚项能够让选中的判别器 $D$ 倾向于是一个“对输入梯度小于 1 的函数”。这样也能实现类似 weight clipping 的效果。  
然而我们无法对所有的 $x$ 求积分，所以用采样的方法来实现这个惩罚项：
$$
\begin{split}
W(P_{data},P_G)=&\mathop{max}_{D}\lbrace E_{x\sim P_{data}}[f(x)]-E_{x\sim P_G}[f(x)]\\
&-\lambda E_{x\sim penalty}[max(0,||\nabla_xD(x)||-1)]\rbrace
\end{split}
$$   
在训练过程中，我们更倾向于得到一个判别器 $D$，它能对从 $P_{penalty}$ 中采样得到的每一个 $x$ 都满足 $||\nabla_xD(x)||\leq1$   
Improved WGAN 设计了一个特别的 $P_{penalty}$，它的产生过程如下:  
1. 从 $P_{data}$ 中采样一个点
2. 从 $P_G$ 中采样一个点
3. 将这两个点连线
4. 在连线之上在采样得到一个点，就是一个从 $P_{penalty}$ 采样的一个点

重复上面的过程就能不断采样得到 $x\sim P_{penalty}$，最终得到下图中的蓝色区域就可以看作是 $P_{penalty}$：  
![](/assets/blog_images/2018-11-27/ImpWGAN1.png)  
Improved WGAN 真正做的是不是要 $||\nabla_xD(x)||\leq1$ 就可以了，而是让 $||\nabla_xD(x)||$ 尽量地接近 1：  
$$
\begin{split}
W(P_{data},P_G)=&\mathop{max}_{D}\lbrace E_{x\sim P_{data}}[f(x)]-E_{x\sim P_G}[f(x)]\\
&-\lambda E_{x\sim penalty}[(||\nabla_xD(x)||-1)^2]\rbrace
\end{split}
$$   
因为一个“好”的判别器应该在 $P_{data}$ 附近尽可能大，要在 $P_G$ 附近尽可能小。也就是说处于 $P_{data}$ 和 $P_G$ 之间的 $P_{penalty}$ 区域应该有一个比较“陡峭”的梯度。但是这个陡峭程度是有限制的，这个限制就是 1。
实验结果也表明WGAN-GP要优于初始的WGAN，例如WGAN-GP的权重值明显正常的分布。  
![](/assets/blog_images/2018-11-27/ImpWGAN2.png)  










