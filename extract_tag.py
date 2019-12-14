from underthesea import word_tokenize, sent_tokenize
from rake_nltk import Rake, Metric

import string
from itertools import chain


class RakeVietNamese(Rake):

    def __init__(
        self,
        stopwordspath=None,
        punctuations=None,
        ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,
        max_length=100000,
        min_length=1,
    ):

        # By default use degree to frequency ratio as the metric.
        if isinstance(ranking_metric, Metric):
            self.metric = ranking_metric
        else:
            self.metric = Metric.DEGREE_TO_FREQUENCY_RATIO

        # vietnamese stopwords
        if stopwordspath:
            with open(stopwordspath,  encoding="utf8") as f:
                self.stopwords = f.read().splitlines()
        else:
            raise Exception('Missing stopwords')

        # If punctuations are not provided we ignore all punctuation symbols.
        self.punctuations = punctuations
        if self.punctuations is None:
            self.punctuations = string.punctuation

        # All things which act as sentence breaks during keyword extraction.
        self.to_ignore = set(chain(self.stopwords, self.punctuations))

        # Assign min or max length to the attributes
        self.min_length = min_length
        self.max_length = max_length

        # Stuff to be extracted from the provided text.
        self.frequency_dist = None
        self.degree = None
        self.rank_list = None
        self.ranked_phrases = None

    def extract_keywords_from_text(self, text):
        sentences = sent_tokenize(text)
        self.extract_keywords_from_sentences(sentences)

    def extract_keywords_from_sentences(self, sentences):
        phrase_list = self._generate_phrases(sentences)
        self._build_frequency_dist(phrase_list)
        self._build_word_co_occurance_graph(phrase_list)
        self._build_ranklist(phrase_list)

    def _generate_phrases(self, sentences):
        phrase_list = set()
        # Create contender phrases from sentences.
        for sentence in sentences:
            word_list = [word.lower() for word in word_tokenize(sentence)]
            phrase_list.update(self._get_phrase_list_from_words(word_list))
        return phrase_list


if __name__ == '__main__':
    # text = """
    # TÂY DU KÝ TRỞ LẠI VỚI PHIÊN BẢN MỸ - LỤC TIỂU LINH ĐỒNG TIẾP TỤC ĐẢM NHIỆM VAI TÔN NGỘ KHÔNG Ở ĐỘ TUỔI 60.
    # Tác phẩm phim điện ảnh mang tên Tây du ký: Mỹ Hầu Vương thật giả (西游记真假美猴王) do Hàn Đình làm đạo diễn đã chính thức bấm máy. Bộ phim này thật chất là Dám hỏi đường đi ở nơi nào (敢问路在何方) và cũng là tác phẩm Tây du ký do Mỹ - Trung hợp tác quay hình.
    # Được biết, bộ phim điện ảnh này đã mời hai diễn viên Lục Tiểu Linh Đồng và Mã Đức Hoa để đảm nhận vai Tôn Ngộ Không và Trư Bát Giới. Trước mắt, phiên bản điện ảnh chỉ mới công bố tên của hai diễn viên này, còn vị trí của các nhân vật khác vẫn chưa tìm được người thích hợp. Đạo diện Hàn Đình còn đăng bài viết trong vòng bạn bè để tìm kiếm diễn viên thích hợp.
    # Vì để đóng bộ phim Tây du ký: Mỹ Hầu Vương thật giả này mà Lục Tiểu Linh Đồng đã sửa kịch bản đến 3 lần.
    # """
    text = """
    Nhắc đến Tết không thể không nhắc tới hoa đào, hoa mai, quất cảnh. Tuy nhiên, thường vốn bỏ ra để kinh doanh các loại cây hoa này khá cao. Thế niên, chiến lược kinh doanh của chị Thu Giang, một giáo viên tiểu học tại Hà Nội là bán các loại hoa như hồng, cúc, ly, dơn, đồng tiền… với số vốn ít nhưng dễ tiêu thụ nên vẫn mang đến khả năng sinh lời cao.
    """
    r = RakeVietNamese(stopwordspath='vietnamese-stopwords.txt')
    r.extract_keywords_from_text(text)
    result = r.get_ranked_phrases_with_scores()
    from pprint import pprint
    pprint(result)
