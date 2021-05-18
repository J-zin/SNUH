from model.SNUH import SNUH

if __name__ == "__main__":
    argparser = SNUH.get_model_specific_argparser()
    hparams = argparser.parse_args()
    model = SNUH(hparams)
    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        # case study
        if model.hparams.data_path == 'data/ng20.tfidf.mat':
            model.hash_codes_visulization()

        if model.hparams.data_path == 'ng20text.tfidf.mat':
            # model.retrival_case_study()
            model.word_embedding_case_study()

        val_perf, test_perf = model.run_test()
        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))