FROM registry.gitlab.com/mlpds_mit/askcos/askcos-base/torchserve:0.2.1

COPY --chown=askcos:askcos ./pathway-ranker.mar /home/askcos/model_store/pathway-ranker.mar
