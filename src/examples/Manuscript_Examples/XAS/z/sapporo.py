from pyscf import gto


# Basis set: Sapporo-QZP-2012-diffuse
sapporo_c = gto.basis.parse('''
BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (13s,9p,5d,4f,3g) -> [8s,6p,4d,3f,2g]
C    S
   6328.5636600              0.0018240
    949.4439690              0.0140650
    216.0721750              0.0715360
     61.0654920              0.2701140
     19.7008680              0.7277000
C    S
      6.9353910              1.0000000
C    S
      2.5266950              1.0000000
C    S
      0.9742940              1.0000000
C    S
      5.0024820             -0.2058730
      0.5285790              1.0757940
C    S
      0.2222920              1.0000000
C    S
      0.1026000              1.0000000
C    S
      0.0342000              1.0000000
C    P
     13.9394660              1.0000000
C    P
     61.6345450              0.0082490
     14.4423380              0.0603590
      4.4412960              0.2583920
      1.5730260              0.7701760
C    P
      0.5979420              1.0000000
C    P
      0.2298670              1.0000000
C    P
      0.0864370              1.0000000
C    P
      0.0288123              1.0000000
C    D
      5.7945660              0.0669920
      1.5521940              0.9653563
C    D
      0.6188830              1.0000000
C    D
      0.2296760              1.0000000
C    D
      0.0765587              1.0000000
C    F
      2.2894950              0.1622509
      1.0747640              0.8753466
C    F
      0.4405970              1.0000000
C    F
      0.1468657              1.0000000
C    G
      1.3396820              0.4982008
      0.5429150              0.6246047
C    G
      0.1879282              1.0000000
END''')
sapporo_o = gto.basis.parse('''
BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (13s,9p,5d,4f,3g) -> [8s,6p,4d,3f,2g]
O    S
  11371.8294630              0.0018030
   1706.0929020              0.0139180
    388.2679950              0.0708700
    109.7507550              0.2687970
     35.4659020              0.7292380
O    S
     12.5340490              1.0000000
O    S
      4.5888600              1.0000000
O    S
      1.9723430              1.0000000
O    S
      9.8428230             -0.2058020
      1.0495500              1.0763360
O    S
      0.4326590              1.0000000
O    S
      0.1916280              1.0000000
O    S
      0.0638760              1.0000000
O    P
     25.6085570              1.0000000
O    P
    111.6435970              0.0083240
     26.2290360              0.0623210
      8.1625280              0.2696390
      2.9239230              0.7578720
O    P
      1.1057210              1.0000000
O    P
      0.4136360              1.0000000
O    P
      0.1479770              1.0000000
O    P
      0.0493257              1.0000000
O    D
     10.5203680              0.0832270
      2.9714730              0.9543244
O    D
      1.1683940              1.0000000
O    D
      0.4268190              1.0000000
O    D
      0.1422730              1.0000000
O    F
      4.8912290              0.1694800
      1.9686530              0.8834389
O    F
      0.7476440              1.0000000
O    F
      0.2492147              1.0000000
O    G
      2.6377970              0.4547998
      0.9721190              0.6857437
O    G
      0.3318957              1.0000000
END''')
