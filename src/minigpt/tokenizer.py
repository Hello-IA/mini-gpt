import regex as re
K = 300
def formatage(texte):
    PATTERN = r"[ldnjmtscLDNJMTSC]['’]| ?\p{L}+| ?\p{N}+| ?[,;:!?\"'’«»()\-]|\.+|\s+|."

    texte =  re.findall(PATTERN, texte)
    
    for i in range(len(texte)):
        texte[i] = texte[i].encode("utf-8")
    return texte
data = open("le_petit_prince.txt", "r")
data = data.read()
segments = formatage(data)


def C_k(segments):
    dic = {}
    for seg in segments:
        for i in range(1, len(seg)):
            if dic.get((seg[i-1],seg[i])) is not None:
                dic[(seg[i-1], seg[i])] +=1
            else:
                dic[(seg[i-1], seg[i])] = 1
    return max(dic, key=dic.get)
#print(C_k(segments))   
def merge(seg, pair, z):
    new_seg = []
    i = 0
    while i < len(seg):
        if i < len(seg)-1 and (seg[i], seg[i+1]) == pair:
            new_seg.append(z)
            i +=2
        else:
            new_seg.append(seg[i])
            i +=1
    return new_seg
 
def BPE(segments):
    regel_dic = {}
    regel = []
    for V_i in range(256, 256+K):
        max_ocur = C_k(segments)
        for i in range(len(segments)):
            segments[i] = merge(segments[i], max_ocur, V_i) 
        regel.append(max_ocur)
    return regel

def encoder(texte, regel):
    
    for i in range(K):
        for j in range(len(texte)):
            texte[j] = merge(texte[j], regel[i][1], regel[i][0])
            
    return texte
def deplis_recurtions(number, regel):
    if number <=255:
        return [number]
    a, b = regel[number-256]
    return deplis_recurtions(a, regel) + deplis_recurtions(b, regel)

def decode(texte, deplis):
    new_texte = []
    for tok in texte:
        for t in tok:
            if t>255:
               new_texte += deplis[t-256]
            else:
                new_texte += [t]
    return bytes(new_texte).decode("utf-8")
regel = BPE(segments)
regel_deplis = []
print("regel :", regel)
for i in range(256, 256+K):
    regel_deplis.append(deplis_recurtions(i, regel))
encodage = encoder(formatage("Je suis milos le chas de maxime."), regel)
print("Je suis milos le chas de maxime.", encodage)
print("deplis ", regel_deplis)
print("decodage", decode(encodage, regel_deplis))
print("formatage :", formatage("Je suis milos le chas de maxime."))
print(decode(encodage, regel_deplis) == "Je suis milos le chas de maxime.")
