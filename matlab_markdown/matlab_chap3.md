# Chapter3

```matlab:Code
clear
```

# 伝達関数モデルの記述

```matlab:Code
Np = [0, 1];      % 伝達関数の分子多項式の係数 (0*s + 1)
Dp = [1, 2, 3];   % 伝達関数の分母多項式の係数 (1*s^2 + 2*s + 3)
P = tf(Np, Dp)
```


```text:Output
P =
 
        1
  -------------
  s^2 + 2 s + 3
 
連続時間の伝達関数です。
```


```matlab:Code
P = tf([0, 1], [1, 2, 3])
```


```text:Output
P =
 
        1
  -------------
  s^2 + 2 s + 3
 
連続時間の伝達関数です。
```

## 練習問題

```matlab:Code
P = tf([1, 2], [1, 5, 3, 4])
```


```text:Output
P =
 
          s + 2
  ---------------------
  s^3 + 5 s^2 + 3 s + 4
 
連続時間の伝達関数です。
```



分母多項式の展開



```matlab:Code
syms s
expand( (s+1)*(s+2)^2 ) 
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;s^3&space;+5\,s^2&space;+8\,s+4"/>

```matlab:Code
P = tf([1, 3],[1, 5, 8, 4])
```


```text:Output
P =
 
          s + 3
  ---------------------
  s^3 + 5 s^2 + 8 s + 4
 
連続時間の伝達関数です。
```


```matlab:Code
P1 = tf([1, 3], [0, 1]);
P2 = tf([0, 1], [1, 1]);
P3 = tf([0, 1], [1, 2]);
P = P1 * P2 * P3^2
```


```text:Output
P =
 
          s + 3
  ---------------------
  s^3 + 5 s^2 + 8 s + 4
 
連続時間の伝達関数です。
```

## 分母・分子多項式の係数の抽出

```matlab:Code
[numP, denP] = tfdata(P, 'v');
numP
```


```text:Output
numP = 1x4    
     0     0     1     3

```


```matlab:Code
denP
```


```text:Output
denP = 1x4    
     1     5     8     4

```

# 状態空間モデルの記述

```matlab:Code
A = [0 1; -1 -1];
B = [0; 1];
C = [1 0];
D = 0;
P = ss(A, B, C, D)
```


```text:Output
P =
 
  A = 
       x1  x2
   x1   0   1
   x2  -1  -1
 
  B = 
       u1
   x1   0
   x2   1
 
  C = 
       x1  x2
   y1   1   0
 
  D = 
       u1
   y1   0
 
連続時間状態空間モデル。
```

## 練習問題

```matlab:Code
A = [1 1 2; 2 1 1; 3 4 5];
B = [2; 0; 1];
C = [1 1 0];
D = 0;
P = ss(A, B, C, D)
```


```text:Output
P =
 
  A = 
       x1  x2  x3
   x1   1   1   2
   x2   2   1   1
   x3   3   4   5
 
  B = 
       u1
   x1   2
   x2   0
   x3   1
 
  C = 
       x1  x2  x3
   y1   1   1   0
 
  D = 
       u1
   y1   0
 
連続時間状態空間モデル。
```

## A,B,C,D行列の抽出

```matlab:Code
[sysA, sysB, sysC, sysD] = ssdata(P)
```


```text:Output
sysA = 3x3    
     1     1     2
     2     1     1
     3     4     5

sysB = 3x1    
     2
     0
     1

sysC = 1x3    
     1     1     0

sysD = 0
```

# ブロック線図の結合

```matlab:Code
S1 = tf( [0, 1], [1, 1])
```


```text:Output
S1 =
 
    1
  -----
  s + 1
 
連続時間の伝達関数です。
```


```matlab:Code
S2 = tf( [1, 1], [1, 1, 1])
```


```text:Output
S2 =
 
     s + 1
  -----------
  s^2 + s + 1
 
連続時間の伝達関数です。
```

## 直列結合

```matlab:Code
S = S2 * S1
```


```text:Output
S =
 
          s + 1
  ---------------------
  s^3 + 2 s^2 + 2 s + 1
 
連続時間の伝達関数です。
```


```matlab:Code

series(S1, S2)
```


```text:Output
ans =
 
          s + 1
  ---------------------
  s^3 + 2 s^2 + 2 s + 1
 
連続時間の伝達関数です。
```



分母分子の共通因子 s+1 が約分されない この場合は，minreal を使う



```matlab:Code
minreal(S)
```


```text:Output
ans =
 
       1
  -----------
  s^2 + s + 1
 
連続時間の伝達関数です。
```

## 並列結合

```matlab:Code
S = S1 + S2
```


```text:Output
S =
 
     2 s^2 + 3 s + 2
  ---------------------
  s^3 + 2 s^2 + 2 s + 1
 
連続時間の伝達関数です。
```


```matlab:Code

S = parallel(S1, S2)
```


```text:Output
S =
 
     2 s^2 + 3 s + 2
  ---------------------
  s^3 + 2 s^2 + 2 s + 1
 
連続時間の伝達関数です。
```

## フィードバック結合

```matlab:Code
S = S1*S2 / (1 + S1*S2)
```


```text:Output
S =
 
           s^4 + 3 s^3 + 4 s^2 + 3 s + 1
  -----------------------------------------------
  s^6 + 4 s^5 + 9 s^4 + 13 s^3 + 12 s^2 + 7 s + 2
 
連続時間の伝達関数です。
```


```matlab:Code

S = feedback(S1*S2, 1)
```


```text:Output
S =
 
          s + 1
  ---------------------
  s^3 + 2 s^2 + 3 s + 2
 
連続時間の伝達関数です。
```


```matlab:Code

minreal(S)
```


```text:Output
ans =
 
       1
  -----------
  s^2 + s + 2
 
連続時間の伝達関数です。
```



ポジティブフィードバックの場合



```matlab:Code
S = feedback(S1*S2, 1, 1)
```


```text:Output
S =
 
       s + 1
  ---------------
  s^3 + 2 s^2 + s
 
連続時間の伝達関数です。
```


```matlab:Code
minreal(S)
```


```text:Output
ans =
 
     1
  -------
  s^2 + s
 
連続時間の伝達関数です。
```

## 練習問題

```matlab:Code
S1 = tf(1, [1, 1]);
S2 = tf(1, [1, 2]);
S3 = tf([3, 1], [1, 0]);
S4 = tf([2, 0], [0, 1]);

S12 = feedback(S1, S2);
S123 = series(S12, S3);
S = feedback(S123, S4)
```


```text:Output
S =
 
    3 s^2 + 7 s + 2
  --------------------
  7 s^3 + 17 s^2 + 7 s
 
連続時間の伝達関数です。
```

# 補遺（実現問題とプロパー性）
## 実現問題

```matlab:Code
P = tf( [0, 1], [1, 1, 1]);

Pss = ss(P)   % 伝達関数モデルから状態空間モデルへの変換
```


```text:Output
Pss =
 
  A = 
       x1  x2
   x1  -1  -1
   x2   1   0
 
  B = 
       u1
   x1   1
   x2   0
 
  C = 
       x1  x2
   y1   0   1
 
  D = 
       u1
   y1   0
 
連続時間状態空間モデル。
```


```matlab:Code
Ptf = tf(Pss) % 状態空間モデルから伝達関数モデルへの変換
```


```text:Output
Ptf =
 
       1
  -----------
  s^2 + s + 1
 
連続時間の伝達関数です。
```

## **可制御正準形**

```matlab:Code
A = [1 2 3; 3 2 1; 4 5 0];
B = [1; 0; 1];
C = [0 2 1];
D = 0;
P = ss(A, B, C, D);

[~,Dp] = tfdata( tf(P), 'v');
W = hankel(fliplr(Dp(1:numel(Dp)-1)));
Uc = ctrb(P.A, P.B);
S = Uc * W;
ss( S\P.A*S, S\P.B, P.C*inv(S), P.D)
```


```text:Output
ans =
 
  A = 
              x1         x2         x3
   x1  4.441e-16          1          0
   x2  8.882e-16  4.441e-16          1
   x3         24         21          3
 
  B = 
       u1
   x1   0
   x2   0
   x3   1
 
  C = 
         x1    x2    x3
   y1   0.5  0.25   0.5
 
  D = 
       u1
   y1   0
 
連続時間状態空間モデル。
```

## 可観測正準形

```matlab:Code
A = [1 2 3; 3 2 1; 4 5 0];
B = [1; 0; 1];
C = [0 2 1];
D = 0;
P = ss(A, B, C, D);

[Np,Dp] = tfdata( tf(P), 'v');
W = hankel(fliplr(Dp(1:numel(Dp)-1)));
Uo = obsv(P.A, P.C);
S = W * Uo;
ss(S*P.A/S, S*P.B, P.C/S, P.D)
```


```text:Output
ans =
 
  A = 
               x1          x2          x3
   x1   7.819e-16  -3.305e-15          24
   x2           1  -8.653e-17          21
   x3           0           1           3
 
  B = 
       u1
   x1  27
   x2   9
   x3   1
 
  C = 
               x1          x2          x3
   y1  -1.742e-17   2.612e-17           1
 
  D = 
       u1
   y1   0
 
連続時間状態空間モデル。
```

## プロパー性

```matlab:Code
S1 = tf([1, 1], [0, 1]);
S2 = tf([0, 1], [1, 1]);

S = minreal(series(S1, S2))
```


```text:Output
S =
 
  1
 
静的ゲインです。
```

  

```matlab:Code
S2
```


```text:Output
S2 =
 
    1
  -----
  s + 1
 
連続時間の伝達関数です。
```


```matlab:Code
ss(S2)
```


```text:Output
ans =
 
  A = 
       x1
   x1  -1
 
  B = 
       u1
   x1   1
 
  C = 
       x1
   y1   1
 
  D = 
       u1
   y1   0
 
連続時間状態空間モデル。
```

  

```matlab:Code
S1
```


```text:Output
S1 =
 
  s + 1
 
連続時間の伝達関数です。
```


```matlab:Code
ss(S1)
```


```text:Output
ans =
 
  A = 
       x1  x2
   x1   1   0
   x2   0   1
 
  B = 
       u1
   x1   0
   x2  -2
 
  C = 
        x1   x2
   y1  0.5  0.5
 
  D = 
       u1
   y1   0
 
  E = 
       x1  x2
   x1   0   1
   x2   0   0
 
連続時間状態空間モデル。
```

# ラプラス変換

```matlab:Code
syms s
syms t positive
laplace(1, t, s)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;\frac{1}{s}"/>

```matlab:Code
laplace(t, t, s)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;\frac{1}{s^2&space;}"/>
  

```matlab:Code
syms a
laplace(exp(-a*t), t, s)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;\frac{1}{a+s}"/>
  

```matlab:Code
syms w positive
laplace(sin(w*t), t, s)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;\frac{w}{s^2&space;+w^2&space;}"/>

```matlab:Code

laplace(cos(w*t), t, s)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;\frac{s}{s^2&space;+w^2&space;}"/>
  

```matlab:Code
laplace(exp(-a*t)*sin(w*t), t, s)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;\frac{w}{{{\left(a+s\right)}}^2&space;+w^2&space;}"/>

```matlab:Code
laplace(exp(-a*t)*cos(w*t), t, s)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;\frac{a+s}{{{\left(a+s\right)}}^2&space;+w^2&space;}"/>
# 逆ラプラス変換

```matlab:Code
ilaplace(1/s, s, t)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;1"/>

```matlab:Code
ilaplace(1/s^2, s, t)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;t"/>

```matlab:Code
ilaplace(1/(s+a), s, t)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;{\mathrm{e}}^{-a\,t}&space;"/>

```matlab:Code
ilaplace( w/(s^2+w^2), s, t)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;\sin&space;\left(t\,w\right)"/>

```matlab:Code
ilaplace( s/(s^2+w^2), s, t)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;\cos&space;\left(t\,w\right)"/>

```matlab:Code
ilaplace( w/((s+a)^2+w^2), s, t)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;{\mathrm{e}}^{-a\,t}&space;\,\sin&space;\left(t\,w\right)"/>

```matlab:Code
ilaplace( (s+a)/((s+a)^2+w^2), s, t)
```

ans = 

   <img src="https://latex.codecogs.com/gif.latex?&space;{\mathrm{e}}^{-a\,t}&space;\,\cos&space;\left(t\,w\right)"/>
