import inltk.inltk
import nltk
import speech_recognition as sr
from nltk import pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

recog = sr.Recognizer()
with sr.Microphone() as source:
    #audio = recog.listen(source, timeout=5, phrase_time_limit=100)
    print("Adjusting noise..")
    recog.adjust_for_ambient_noise(source, duration=0.5)
    print("Listening..")
    audio_input = recog.listen(source)
    try:
        print('Done, Please wait while we are processing what you said...')
        text = recog.recognize_google(audio_input)
        print("Response: {}".format(text))


        #filter = re.search(r"(?:Mr|Ms) (\S+)\s+(.\S+)", text)
        #target = filter.groups()


        #testtext = "Hello, Where is Margav Ghoghari"
        contentArray = [text]
        try:
            for item in contentArray:
                stop_words = set(stopwords.words('hindi'))
                word_tokens = word_tokenize(item)
                filtered_sentence = [w for w in word_tokens if not w in stop_words]
                print("word_tokens: ", word_tokens)
                print("filtered_sentence: ", filtered_sentence)

                tagged = nltk.pos_tag(filtered_sentence)
                print("tagged: ", tagged)

                ne_tree = ne_chunk(pos_tag(word_tokenize(text)))
                print("ne_tree:", ne_tree)

                iob_tagged = tree2conlltags(ne_tree)
                print("iob_tagged:", iob_tagged)

                ne_tree = conlltags2tree(iob_tagged)
                print("ne_tree:", ne_tree)

                ne_tree.draw()

        except Exception as err:
            print(f"Error: {err}")

    except sr.UnknownValueError:
        print("Audio not understandable.")
    except sr.RequestError as e:
        print("Request Error: {0}".format(e))
    except Exception as error:
        print(f"Error: {error}")