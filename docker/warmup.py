from transformers import (EncoderDecoderModel,
                          PreTrainedModel,
                          BertTokenizer,
                          BertGenerationEncoder,
                          BertGenerationDecoder)

encoder = BertGenerationEncoder.from_pretrained(model_type,
                                                bos_token_id=BOS_TOKEN_ID,
                                                eos_token_id=EOS_TOKEN_ID) # add cross attention layers and use BERT’s cls token as BOS token and sep token as EOS token

decoder = BertGenerationDecoder.from_pretrained(model_type,
                                                add_cross_attention=True,
                                                is_decoder=True,
                                                bos_token_id=BOS_TOKEN_ID,
                                                eos_token_id=EOS_TOKEN_ID)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to(device)
