from src.app import InputData
from locust import HttpUser, task, between
import json


class PerformanceTest(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def test_prob1(self):
        sample = InputData(
            id='test',
            columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6',
                     'feature7', 'feature8', 'feature9', 'feature10', 'feature11',
                     'feature12', 'feature13', 'feature14', 'feature15', 'feature16'],
            rows=[['Site engineer',
                   'grocery_pos',
                   8.6,
                   48230,
                   40.21343879339888,
                   -85.2037563034635,
                   47583,
                   42.508293,
                   -83.168004,
                   65.59606217585437,
                   3,
                   5,
                   1,
                   8.017864754614141,
                   1.0288222577545105,
                   58.91113204988728]] * 1000)
        headers = {'Accept': 'application/json',
                   'Content-Type': 'application/json'}
        res = self.client.post("/phase-1/prob-1/predict",
                               data=json.dumps(sample.dict()),
                               headers=headers)

    # @task(2)
    # def test_prob2(self):
    #     sample = InputData(
    #         id='test2',
    #         columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6',
    #                  'feature7', 'feature8', 'feature9', 'feature10', 'feature11',
    #                  'feature12', 'feature13', 'feature14', 'feature15', 'feature16',
    #                  'feature17', 'feature18', 'feature19', 'feature20'],
    #         rows=[['V1',
    #                 4.781942,
    #                 'V2',
    #                 'V8',
    #                 1337.025331,
    #                 'V4',
    #                 'V4',
    #                 2.0,
    #                 'V2',
    #                 'V0',
    #                 4.0,
    #                 'V2',
    #                 35.689494,
    #                 'none',
    #                 'V1',
    #                 1.0,
    #                 'V2',
    #                 1.0,
    #                 'none',
    #                 'yes']]*1000)
    #     headers = {'Accept': 'application/json',
    #                    'Content-Type': 'application/json'}
    #     res = self.client.post("/phase-1/prob-2/predict",
    #                            data=json.dumps(sample.dict()),
    #                            headers=headers)