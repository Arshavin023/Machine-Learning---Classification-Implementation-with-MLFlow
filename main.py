from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline


STAGE_NAME_01 = 'Data Ingestion stage'

try:
    logger.info(f">>>>>> stage {STAGE_NAME_01} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME_01} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME_02 = 'Data Validation stage'

try:
    logger.info(f">>>>>> stage {STAGE_NAME_02} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME_02} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME_03 = 'Data Transformation stage'

try:
    logger.info(f">>>>>> stage {STAGE_NAME_03} started <<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME_03} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
