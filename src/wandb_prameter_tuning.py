from pyloraks import optimize_loraks_params
import logging


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)

    logging.info(f"_______________________________")
    logging.info(f"___ PyLORAKS reconstruction ___")
    logging.info(f"_______________________________")
    try:
        optimize_loraks_params()
    except Exception as e:
        logging.exception(e)
        exit(-1)
