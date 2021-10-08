import boto3
import json
import sys
import time
import aws_table_to_csv as atc

'''
************************
Code from https://docs.aws.amazon.com/textract/latest/dg/async-analyzing-with-sqs.html
Adapted to handle multiple tables on many pages to a csv file
'''


class ProcessType:
    DETECTION = 1
    ANALYSIS = 2


class DocumentProcessor:
    jobId = ''
    region_name = ''

    roleArn = ''
    bucket = ''
    document = ''

    sqsQueueUrl = ''
    snsTopicArn = ''
    processType = ''

    def __init__(self, role, bucket, document, region):
        self.roleArn = role
        self.bucket = bucket
        self.document = document
        self.region_name = region

        self.textract = boto3.client('textract', region_name=self.region_name)
        self.sqs = boto3.client('sqs')
        self.sns = boto3.client('sns')

    def ProcessDocument(self, type):
        jobFound = False

        self.processType = type
        validType = False

        if self.processType == ProcessType.ANALYSIS:
            response = self.textract.start_document_analysis(
                DocumentLocation={'S3Object': {
                    'Bucket': self.bucket, 'Name': self.document}},
                FeatureTypes=["TABLES"],
                NotificationChannel={'RoleArn': self.roleArn, 'SNSTopicArn': self.snsTopicArn})
            print('Processing type: Analysis')
            validType = True

        if validType == False:
            print("Invalid processing type. Choose Detection or Analysis.")
            return

        print('Start Job Id: ' + response['JobId'])
        dotLine = 0
        while jobFound == False:
            sqsResponse = self.sqs.receive_message(QueueUrl=self.sqsQueueUrl, MessageAttributeNames=['ALL'],
                                                   MaxNumberOfMessages=10)

            if sqsResponse:

                if 'Messages' not in sqsResponse:
                    if dotLine < 40:
                        print('.', end='')
                        dotLine = dotLine + 1
                    else:
                        print()
                        dotLine = 0
                    sys.stdout.flush()
                    time.sleep(5)
                    continue

                for message in sqsResponse['Messages']:
                    notification = json.loads(message['Body'])
                    textMessage = json.loads(notification['Message'])
                    print(textMessage['JobId'])
                    print(textMessage['Status'])
                    if str(textMessage['JobId']) == response['JobId']:
                        print('Matching Job Found:' + textMessage['JobId'])
                        jobFound = True
                        self.GetResults(textMessage['JobId'])
                        self.sqs.delete_message(QueueUrl=self.sqsQueueUrl,
                                                ReceiptHandle=message['ReceiptHandle'])
                    else:
                        print("Job didn't match:" +
                              str(textMessage['JobId']) + ' : ' + str(response['JobId']))
                    # Delete the unknown message. Consider sending to dead letter queue
                    self.sqs.delete_message(QueueUrl=self.sqsQueueUrl,
                                            ReceiptHandle=message['ReceiptHandle'])

        print('Done!')

    def CreateTopicandQueue(self):

        millis = str(int(round(time.time() * 1000)))

        # Create SNS topic
        snsTopicName = "AmazonTextractTopic" + millis

        topicResponse = self.sns.create_topic(Name=snsTopicName)
        self.snsTopicArn = topicResponse['TopicArn']

        # create SQS queue
        sqsQueueName = "AmazonTextractQueue" + millis
        self.sqs.create_queue(QueueName=sqsQueueName)
        self.sqsQueueUrl = self.sqs.get_queue_url(
            QueueName=sqsQueueName)['QueueUrl']

        attribs = self.sqs.get_queue_attributes(QueueUrl=self.sqsQueueUrl,
                                                AttributeNames=['QueueArn'])['Attributes']

        sqsQueueArn = attribs['QueueArn']

        # Subscribe SQS queue to SNS topic
        self.sns.subscribe(
            TopicArn=self.snsTopicArn,
            Protocol='sqs',
            Endpoint=sqsQueueArn)

        # Authorize SNS to write SQS queue
        policy = """{{
            "Version":"2012-10-17",
            "Statement":[
                {{
                "Sid":"MyPolicy",
                "Effect":"Allow",
                "Principal" : {{"AWS" : "*"}},
                "Action":"SQS:SendMessage",
                "Resource": "{}",
                "Condition":{{
                    "ArnEquals":{{
                    "aws:SourceArn": "{}"
                    }}
                }}
                }}
            ]
            }}""".format(sqsQueueArn, self.snsTopicArn)

        response = self.sqs.set_queue_attributes(
            QueueUrl=self.sqsQueueUrl,
            Attributes={
                'Policy': policy
            })

    def DeleteTopicandQueue(self):
        self.sqs.delete_queue(QueueUrl=self.sqsQueueUrl)
        self.sns.delete_topic(TopicArn=self.snsTopicArn)

    def GetResults(self, jobId):
        maxResults = 1000
        paginationToken = None
        finished = False
        loop_num = 0
        entire_response = []

        while finished == False:
            response = None

            if self.processType == ProcessType.ANALYSIS:
                if paginationToken == None:
                    response = self.textract.get_document_analysis(JobId=jobId,
                                                                   MaxResults=maxResults)
                else:
                    response = self.textract.get_document_analysis(JobId=jobId,
                                                                   MaxResults=maxResults,
                                                                   NextToken=paginationToken)

            if self.processType == ProcessType.DETECTION:
                if paginationToken == None:
                    response = self.textract.get_document_text_detection(JobId=jobId,
                                                                         MaxResults=maxResults)
                else:
                    response = self.textract.get_document_text_detection(JobId=jobId,
                                                                         MaxResults=maxResults,
                                                                         NextToken=paginationToken)

            blocks = response['Blocks']
            if loop_num == 0:
                print('Detected Document Text')
                print('Pages: {}'.format(
                    response['DocumentMetadata']['Pages']))
                loop_num += 1
            else:
                print("Response Number: {}".format(str(loop_num)))
                loop_num += 1

            entire_response.extend(blocks)

            with open('test' + str(loop_num) + '.json', 'w') as j_file:
                j_file.write(json.dumps(response))

            if 'NextToken' in response:
                paginationToken = response['NextToken']
            else:
                finished = True
        print("Finished collecting responses...")

        atc.main(self.document, entire_response)


def main(roleArn, bucket, document, region_name):
    roleArn = roleArn
    bucket = bucket
    document = document
    region_name = region_name

    analyzer = DocumentProcessor(roleArn, bucket, document, region_name)
    analyzer.CreateTopicandQueue()
    analyzer.ProcessDocument(ProcessType.ANALYSIS)
    analyzer.DeleteTopicandQueue()


if __name__ == "__main__":
    roleArn = sys.argv[1]
    bucket = sys.argv[2]
    document = sys.argv[3]
    region_name = sys.argv[4]
    main(roleArn, bucket, document, region_name)
