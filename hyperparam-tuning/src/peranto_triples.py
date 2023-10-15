class PerantoTripleStore:
    """
    Basically a TripleStore where you 
    just add data and then are able to read different components of it

    Similar to TripleStore functionality
    """
    def __init__(self):
        self.data = []    
        self.load_from_file()
    
    def add_triple(self, s, v, o,):
        self.data.append({
            'subject'  : s,
            'verb'     : v,
            'object'   : o,
            })
    
    def _retrieve(self, parts):
        return [tuple(item[part] for part in parts if part) for item in self.data]

    # Specific retrieval methods
    def get_triples(self):
        return self._retrieve(['subject', 'verb', 'object'])
    
    def get_sv_pairs(self):
        return self._retrieve(['subject', 'verb'])
    
    def get_vo_pairs(self):
        return self._retrieve(['verb', 'object'])
    
    def get_subjects(self):
        return self._retrieve(['subject'])
    
    def get_verbs(self):
        return self._retrieve(['verb'])
    
    def get_objects(self):
        return self._retrieve(['object'])
    
    def get_nouns(self):
        subjects = self._retrieve(['subject'])
        objects = self._retrieve(['object'])
        return list(set(subjects + objects))
    
    def get(self, distribution):
        if distribution == 'vb':
            return self.get_verbs()

        elif distribution == 'nn':
            return self.get_nouns()

        elif distribution == 'nn.arg0':
            return self.get_subjects()

        elif distribution == 'nn.arg1':
            return self.get_objects()

        elif distribution == "nn.arg0.$y0":
            return self.get_sv_pairs()
        
        elif distribution == 'nn.arg1.$y0':
            return self.get_vo_pairs()

        else:
            raise Exception(f"Please enter a valid distribution. Note that my cousin TripleStore will not be as kind.")
