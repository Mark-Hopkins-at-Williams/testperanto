from testperanto.morphology import Morpher


class JapanesePronounMorpher(Morpher):
    
    def morph(self, word, properties):
        if properties['COUNT'] == "sng":
            if properties['PERSON'] == "1":
                return "watashi"
            elif properties['PERSON'] == "2":
                return "anata"
            elif properties['PERSON'] == "3":
                return "kare"
            else:
                raise Exception(f"Person not recognized: {properties['PERSON']}")
        elif properties['COUNT'] == "plu":
            if properties['PERSON'] == "1":
                return "watashitachi"
            elif properties['PERSON'] == "2":
                return "anata"
            elif properties['PERSON'] == "3":
                return "karera"
            else:
                raise Exception(f"Person not recognized: {properties['PERSON']}")
        else:
            raise Exception(f"Count not recognized: {properties['COUNT']}")
        
