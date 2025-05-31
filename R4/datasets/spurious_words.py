# Taken from https://github.com/tapilab/emnlp-2020-spurious/blob/main/spurious_words.py

__imdb_bad_pos = ['count', 'imax', 'spider', 'ages', 'secretary', 'cinema', 'sexual', 'thornberrys',
               'animated', 'bride', 'pulls', 'history', 'son', 'pray', 'still', 'pacino', 'thanks',
               'department', 'calls', 'vietnam', 'performances', 'spielberg', 'detailed',
               'speaks', 'alan', 'everyday', 'russian', 'martha', 'majidi', 'roles',
               'photo', 'summer', 'profile', 'mike', 'examines', 'reminder', 'political',
               'form', 'bourne', 'popcorn', 'mamet', 'anime', 'am', 'medium', 'captures',
               'distinct', 'stylistic', 'guaranteed', 'smith', 'frailty', 'debate',
               'rhythm', 'spare', 'freedom', 'weeks', 'provides', 'confidence',
               'constructed', 'culture', 'open', 'conduct', 'help', 'grown', 'skin',
               'manages', 'tear']

__imdb_bad_neg = ['pie', 'neither', 'too', 'disguise', 'product', 'tv', 'animal', 'schneider',
           'benigni', 'none', 'sheridan', 'seagal', 'god', 'college', 'demme', 'named', 'six',
           'wilder', 'numbers', 'apparently', 'merit', 'track', 'idea',
           'violence', 'title', 'snake', 'total', 'niro', 'guns', 'somewhere',
           'lust', 'alas', 'textbook', 'showtime', 'car', 'follows', 'quarter',
           'built', 'jonah', 'ballistic', 'sort', 'chosen', 'television', 'shifting',
           'pacing', 'affair', 'guess', 'independent', 'jay', 'evidence',
           'purpose', 'add', 'premise', 'elaborate', 'putting', 'sequences', 'produce', 'sequence',
           'treats', 'john', 'etc', 'instead', 'thousand', 'pinocchio', 'requires',
           'already',  'pulp', 'unintentional', 'unintentionally', 'meant',
           'wannabe', 'unless', 'stunt', 'jokes', 'wasn', 'hasn', 'save', 'bits',
           'heavy']

def get_spurious_words():

    spurious_words_map = {}
    spurious_words_map['imdb_bad_pos'] = __imdb_bad_pos
    spurious_words_map['imdb_bad_neg'] = __imdb_bad_neg

    return spurious_words_map

def all_imdb_spur():
    return __imdb_bad_pos + __imdb_bad_neg