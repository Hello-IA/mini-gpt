import regex as re
import json
import time


class BPETokenizer:
    def __init__(self):
        self.regel = []
        self.regel_deplis = []
        self.K = None
    @classmethod
    def du_fichier(cls, path_data):
        tok = cls()
        with open(path_data, "r", encoding="utf-8") as f:
            data = json.load(f)
            tok.regel = [tuple(p) for p in data["regel"]]
            tok.regel_deplis = data["regel_deplis"]
            tok.K = data["K"]
        return tok
    @classmethod
    def du_texte(cls, path_data, K):
        tok = cls()
        tok.K = K
        with open(path_data, "r", encoding="utf-8") as f:
            tok.data = f.read()
        tok.segments = tok.formatage(tok.data)
        tok.regel = []
        tok.regel_deplis = []
        return tok
    def writh(self, path_save):
        data = {
                "regel": self.regel, 
                "regel_deplis": self.regel_deplis,
                "K": self.K}
        with open(path_save, "w", encoding="utf-8") as f:
            json.dump(data, f)
    def formatage(self, texte):
        PATTERN = r"[ldnjmtscLDNJMTSC]['’]| ?\p{L}+| ?\p{N}+| ?[,;:!?\"'’«»()\-]|\.+|\s+|."

        texte =  re.findall(PATTERN, texte)
    
        for i in range(len(texte)):
            texte[i] = texte[i].encode("utf-8")
        return texte
    def C_k(self):
        dic = {}
        for seg in self.segments:
            for i in range(1, len(seg)):
                if dic.get((seg[i-1],seg[i])) is not None:
                    dic[(seg[i-1], seg[i])] +=1
                else:
                    dic[(seg[i-1], seg[i])] = 1
        return max(dic, key=dic.get)
    def merge(self, seg, pair, z):
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
     
    def BPE(self):
        for V_i in range(256, 256+self.K):
            max_ocur = self.C_k()
            for i in range(len(self.segments)):
                self.segments[i] = self.merge(self.segments[i], max_ocur, V_i) 
            self.regel.append(max_ocur)

    def encoder(self, texte):
        texte = self.formatage(texte)
        for i in range(self.K):
            pair = self.regel[i]       
            z = 256 + i                 # l'id du token créé à cette étape
            for j in range(len(texte)):
                texte[j] = self.merge(texte[j], pair, z)
        return texte
    def deplis_recurtions(self, number):
        if number <=255:
            return [number]
        a, b = self.regel[number-256]
        return self.deplis_recurtions(a) + self.deplis_recurtions(b)
    def deplis_regel(self):
        for i in range(256, 256+self.K):
            self.regel_deplis.append(self.deplis_recurtions(i))

    def decode(self, texte_encoder):
        new_texte = []
        for tok in texte_encoder:
            for t in tok:
                if t>255:
                   new_texte += self.regel_deplis[t-256]
                else:
                    new_texte += [t]
        return bytes(new_texte).decode("utf-8")
if __name__ == "__main__":
    #start = time.perf_counter()
    bpe = BPETokenizer.du_fichier("data/save.json")
    #bpe.BPE()
    with open("data/bpe_test_pieges.txt", "r", encoding="utf-8") as f:
            text = f.read()

    #elapsed = time.perf_counter() - start
    #print(f"Temps pour K=6000 : {elapsed:.1f}s")
    #bpe.deplis_regel()
    #bpe.writh()

    #print(bpe.encoder(text)
    #print(bpe.decode(bpe.encoder("le chas de maxime et millo")))
    
    print(bpe.decode(bpe.encoder(text)) == text)
