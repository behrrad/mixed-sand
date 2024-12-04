import random
import sys
import warnings

from math import factorial

from relationGraph import RelationGraph
import sqlite3, json, math, pickle, os, signal, re, requests
from itertools import combinations
import networkx as nx
import numpy as np
import heapq
import time
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp, zscore
import statistics
from wikimapper import WikiMapper
# from get_dbpedia_data import get_features_and_kinds_multi_thread

from generate_new_columns import k_means, k_means_clustering, normal_k_means_clustering, find_number_of_clusters, find_number_of_clusters_with_normal_k_means


warnings.filterwarnings("ignore", category=FutureWarning)

NUM_CLUSTERS_K_MEANS = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
NUM_CLUSTERS_NORMAL_K_MEANS = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
class TimeoutError(Exception):
    pass


def TimeoutHandler(signum, frame):
    raise TimeoutError


class Annotation():
    def __init__(self):
        self.tables = []
        self.graph = RelationGraph()

        self.conn = None
        self.cursor = None
        self.wdc = 'dataset/wdc.txt'
        self.dbname = 'relation.db'
        # self.dbname = 'relation-dbpedia.db'
        self.getEntityQuery = 'https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids=%s&languages=en'
        self.searchQuery = 'https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search=%s&language=en'

        self.k = 5
        self.number_of_mixed_units_columns = 0
        self.number_of_not_mixed_units_columns = 0

        self.ineligible = 1
        self.wikimapper = WikiMapper("index_enwiki-latest.db")

    def connect(self):
        if self.conn == None:
            self.conn = sqlite3.connect('./' + self.dbname)
            self.cursor = self.conn.cursor()

    def clean(self):
        self.cursor = None
        if self.conn != None:
            self.conn.close()
            self.conn = None

    def __loadTable__(self, table):
        columns = []
        table = json.loads(table)

        return columns

    def resolve(self, symbol, reverse=False):
        if reverse:
            rv = []
            query = '''select wid from mapping where label = ?; '''
            self.cursor.execute(query, (symbol,))
            row = self.cursor.fetchone()
            while row != None:
                rv.append(row[0])
                row = self.cursor.fetchone()
            return rv

        else:
            query = '''select label from mapping where wid=?; '''
            self.cursor.execute(query, (symbol,))
            row = self.cursor.fetchone()
            if row == None:
                return None

            return row[0]

    def resolveTriple(self, triple):
        if len(triple) == 0: return None

        rv = tuple()
        for wid in triple:
            rv += (self.resolve(wid),)

        return rv

    def get_parents_to_the_root(self, graph, wid):
        data = []
        data.extend(wid)
        checked_parents = []
        while len(data) > 0:
            edges = graph.in_edges(data[0])
            for edge in list(edges):
                if list(edge)[0] in checked_parents:
                    continue
                checked_parents.append(list(edge)[0])
            data = list(dict.fromkeys(data))
            # checked_parents.append(data[0])
            del data[0]
        return checked_parents

    def createOrder(self, query, dataColumns):
        # create an ordering of processing based on estimated lower bound cost
        queue = []
        topK = []
        for i in range(self.k):
            heapq.heappush(topK, (np.inf, tuple()))

        for item in dataColumns:
            key = item[0]
            column = item[1]
            # NEW CODE
            if len(column) < 3: continue
            # if len(query) > len(column): continue
            cost = self.estimateByRange(query, column)

            try:
                heapq.heappush(queue, (cost, key,))
            except:
                continue
        return queue, topK

    def estimateByRange(self, query, data):
        rd = (min(data), max(data))
        rq = (min(query), max(query))

        cost = 0

        # query column is within data column
        if (rq[0] > rd[0] and rq[1] < rd[1]):
            cost = 0
        # entire query column is less than data column
        elif (rq[1] < rd[0]):
            for num in rq:
                cost += abs(rd[0] - num)
        # entire query column is greaater than data column
        elif (rq[0] > rd[1]):
            for num in rq:
                cost += abs(num - rd[1])

        # there exist partial overlap between columns
        # map non overlap numbers to the minimum value
        elif (rq[0] < rd[0]):
            for num in rq:
                if num < rd[0]: cost += abs(num - rd[0])
        # map non overlap numbers to the maximum value
        elif (rq[1] > rd[1]):
            for num in rq:
                if num > rd[1]: cost += abs(num - rd[1])
        # new code
        if len(query) > len(data):
            cost = cost * min(50, len(query)) / min(50, len(data))
        return cost

    def pruneUp(self, query, data, topK):
        # if the estimated lower bound is greater than the largest cost in topK, we can prune this column
        if len(data) < 3:
            return -1
        est = self.estimateByRange(query, data)

        currentMax = topK[-1][0]  # heapq.nlargest(1, self.topK, key=lambda x:x[0])

        return est

    def pruneDown(self, query, data, topK):
        # find lower bound by mapping each element from query to the nearest element in data
        data = np.array(data)
        currentMax = topK[-1][0]

        est = 0
        for num in query:
            est += np.min(np.abs(data - num))
        # new code
        if len(query) > len(data):
            est = est * min(len(query), 50) / min(len(data), 50)
        return est

    def testReduction(self):
        '''
        test for 
        (1): when the query columns is large (larger than some threshold), pick n sample subsets and
             compare the precision when varying n.

        (2): when |q| > |c|, allowing |q|/|c| replacements for each element.
            partition the knowledge columns at ratio 4:6, and pick the larger columns as query column.
        '''

        with open('reformed.pkl', 'rb') as f:
            allColumns = pickle.load(f)

        query = {}
        data = {}
        ratio = 0.2
        for eType in allColumns:
            query[eType] = {}
            data[eType] = {}
            for prop in allColumns[eType]:
                v = allColumns[eType][prop]
                if len(v) < 100 or len(v) > 1000: continue
                qsize = math.ceil(ratio * len(v))

                np.random.shuffle(v)
                query[eType][prop] = v[:qsize]
                data[eType][prop] = v[qsize:]

        allTypes = list(allColumns.keys())
        ind = np.arange(len(allColumns))
        np.random.shuffle(ind)

        samplesize = 0.1
        numSamples = np.arange(1, 11)
        total = [0 for x in range(len(numSamples))]
        correct = [0 for x in range(len(numSamples))]
        times = [0 for x in range(len(numSamples))]

        m = 0
        for i in ind[:200]:
            print(m)
            m += 1
            t = allTypes[i]
            if len(query[t]) == 1 or len(query[t]) == 0: continue

            columns = list(query[t].items())
            np.random.shuffle(columns)
            q = columns[0]

            label, values = q[0], q[1]
            np.random.shuffle(values)
            for k, s in enumerate(numSamples):
                s = max(math.ceil(len(q[1]) * samplesize * s), len(q[1]))

                test = q[1][:s]

                scores = []
                for pr in data[t]:
                    try:
                        start = time.time()
                        cost = self.computeCost(test, data[t][pr])
                        end = time.time()
                        scores.append([pr, cost])
                        # times[k] += (end - start)
                        # total[k] += 1
                    except:
                        continue
                    scores = sorted(scores, key=lambda x: x[1])
                    prediction = scores[0][0]

                    if prediction == label:
                        correct[k] += 1
                    total[k] += 1

        print("k \t correct \t total \t precision")
        for k in range(len(numSamples)):
            print(numSamples[k], correct[k], total[k], correct[k] / total[k])

    # test (1) |q| > |c|
    def testReduction2(self):
        ratio = 0.6
        with open('reformed.pkl', 'rb') as f:
            allColumns = pickle.load(f)

        query = {}
        data = {}
        for eType in allColumns:
            query[eType] = {}
            data[eType] = {}
            for prop in allColumns[eType]:
                v = allColumns[eType][prop]
                if len(v) > 1000 or len(v) < 10: continue
                np.random.shuffle(v)

                qsize = int(len(v) * ratio)
                query[eType][prop] = v[:qsize]
                data[eType][prop] = v[qsize:]

        allTypes = list(query.keys())
        np.random.shuffle(allTypes)
        offsets = [1.0, 1.5, 2.0, 2.5, 'unlimited']
        corrects = [0 for x in offsets]
        total = [0 for x in offsets]
        m = 0
        for eType in allTypes[:500]:
            m += 1
            print(m)
            for i, offset in enumerate(offsets):
                if len(query[eType]) == 1 or len(query[eType]) == 0: continue

                columns = list(query[eType].items())
                np.random.shuffle(columns)
                q = columns[0]
                label, values = q[0], q[1]
                scores = []

                if offset == 'unlimited':
                    for pr in data[eType]:
                        cost = 0
                        for num in values:
                            cost += np.min(np.abs(np.array(data[eType][pr]) - num))
                        scores.append([pr, cost])

                else:
                    for pr in data[eType]:
                        baseReplace = math.ceil(len(q[1]) / len(data[eType][pr]) * offset)
                        # baseReplace = min(2 * baseReplace, baseReplace + offset)
                        datacolumn = []
                        for x in range(baseReplace):
                            datacolumn.extend(data[eType][pr])
                        # print(query[key], datacolumn)
                        cost = self.computeCost(values, datacolumn)
                        scores.append([pr, cost])

                scores = sorted(scores, key=lambda x: x[1])
                if label == scores[0][0]:
                    corrects[i] += 1

                total[i] += 1

        print("offset, corrects, total, precision")
        for i in range(len(offsets)):
            print(offsets[i], corrects[i], total[i], corrects[i] / total[i])

    def testReduction3(self):
        '''
        varying the length of subset chose, and plot against precision, running time
        '''

        with open('reformed.pkl', 'rb') as f:
            allColumns = pickle.load(f)

        query = {}
        data = {}
        ratio = 0.4
        for eType in allColumns:
            query[eType] = {}
            data[eType] = {}
            for prop in allColumns[eType]:
                v = allColumns[eType][prop]
                if len(v) < 50: continue
                qsize = math.ceil(ratio * len(v))

                np.random.shuffle(v)
                query[eType][prop] = v[:qsize]
                data[eType][prop] = v[qsize:]

        allTypes = list(allColumns.keys())
        ind = np.arange(len(allColumns))
        np.random.shuffle(ind)

        sizes = [60]
        sizes = sizes[::-1]
        ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        total = [0 for x in range(len(sizes))]
        correct = [0 for x in range(len(sizes))]
        times = [0 for x in range(len(sizes))]
        np.random.seed(10000)
        print(len(ind))
        m = 0
        for i in ind[:2000]:
            print(m)
            m += 1
            t = allTypes[i]
            if len(query[t]) == 1 or len(query[t]) == 0: continue

            columns = list(query[t].items())
            np.random.shuffle(columns)
            q = columns[0]

            label, values = q[0], q[1]
            np.random.shuffle(values)
            skip = 0
            for k, s in enumerate(sizes):
                # for k, s in enumerate(ratios):
                # s = min(100, math.ceil(s * len(q[1])))
                # if s > 100: continue
                if s <= len(q[1]):
                    test = q[1][:s]

                    scores = []
                    for pr in data[t]:
                        try:
                            start = time.time()
                            cost = self.computeCost(test, data[t][pr])
                            end = time.time()
                            scores.append([pr, cost])
                            times[k] += (end - start)
                            total[k] += 1
                        except:
                            skip = 1
                            break
                    if skip == 1:
                        skip = 0
                        continue
                    scores = sorted(scores, key=lambda x: x[1])
                    prediction = scores[0][0]

                    if prediction == label:
                        correct[k] += 1
                    # total[k] += 1
        print(times, total)
        for i in range(len(times)):
            print(times[i] / total[i])
        # print("k \t correct \t total \t precision")
        # for k in range(len(sizes)):
        #    print(sizes[k], correct[k], total[k], correct[k]/total[k])
        # print (times[k]/total[k], end=',')

    def testefficiency(self):
        self.connect()

        with open('allColumns.pickle', 'rb') as f:
            poplist = []
            dataset = pickle.load(f)
            for key in dataset:
                dataset[key] = np.array(dataset[key])
                if len(dataset[key]) == 1 or len(dataset[key]) > 1000:
                    poplist.append(key)

        for p in poplist:
            dataset.pop(p)

        allcolumns = dataset.items()

        with open(self.wikitable, 'r') as tbf:
            line = tbf.readline().strip()
            for i in range(10):
                pruned1 = 0
                pruned2 = 0
                table = json.loads(line)

                headers = table['header']
                values = table['values']
                entities = table['entity']
                unit = table['unit']

                values = ';'.join(values).replace(',', '').split(';')
                values = list(map(float, values))

                queue, topK = self.createOrder(values, allcolumns)
                print('total columns: ', len(queue))

                while len(queue) != 0:
                    data = heapq.heappop(queue)
                    datacolumn = dataset[data[1]]
                    if self.pruneUp(values, datacolumn, topK):
                        pruned1 += 1

                    elif self.pruneDown(values, datacolumn, topK):
                        pruned2 += 1

                    else:
                        cost = self.computeCost(values, datacolumn)
                        # update top k
                        if cost < topK[-1][0]:
                            topK[-1] = (cost, data[1],)
                            topK.sort(key=lambda x: x[0])

                line = tbf.readline().strip()

                print(headers)
                print('columns pruned by range:', pruned1)
                print('columns pruned by mapping:', pruned2)
                for t in topK:
                    print('predicted label: ', self.resolveTriple(t[1]))
                print('------------------------------------')

    def remove_outliers(self, datacolumn):
        zscores = zscore(datacolumn)
        new_data = []
        for i in range(len(zscores)):
            if 3 > zscores[i] > -3:
                new_data.append(datacolumn[i])
        return new_data

    def update_topk(self, topK, type_property_unit, cost):
        for i in range(len(topK)):
            if topK[i][0] == np.inf:
                break
            # MEETING 12 OCT SHORO
            # if topK[i][1][1] == type_property_unit[1] and topK[i][1][2] == type_property_unit[2]:
            if topK[i][1][1] == type_property_unit[1]:
                if topK[i][0] > cost:
                    topK[i] = (cost, type_property_unit,)
                    topK.sort(key=lambda x: x[0])
                return topK
            # MEETING 12 OCT TAMOM
        topK[-1] = (cost, type_property_unit,)
        topK.sort(key=lambda x: x[0])
        return topK

    def predict_without_order(self, eType, values, distributions, dtype='int'):
        # retrieve labels by mapping function
        # set the threshold for min-cost
        thres = abs(0.3 * sum(values))
        isPattern = False
        # check pre-defined patterns
        if dtype == 'int':
            (isPattern, l) = self.patternMatching(values)

        # TODO: REMOVE PATTERN MATCHING FOR TESTING PURPOSE
        isPattern = False

        allcolumns = distributions.items()

        # pruning
        queue, topK = self.createOrder(values, allcolumns)
        # numCols = len(queue)

        pruned1, pruned2 = 0, 0
        i = 0
        while i < len(list(distributions.keys())):
            # new code
            dict_key = list(distributions.keys())[i]
            datacolumn = distributions[dict_key]
            i += 1
            currentMax = topK[-1][0]
            # new code
            datacolumn = self.remove_outliers(datacolumn)
            est = self.pruneUp(values, datacolumn, topK)
            if est > currentMax or est == -1:
                # new code
                pruned1 += 1

            elif self.pruneDown(values, datacolumn, topK) > currentMax:
                # new code
                pruned2 += 1

            else:
                try:
                    if len(values) < 51:
                        # NEW CODE
                        if len(values) > len(datacolumn):
                            np.random.shuffle(values)
                            cost = self.computeCost(values[:len(datacolumn)], datacolumn)
                            cost = cost * len(values) / len(datacolumn)
                        else:
                            cost = self.computeCost(values, datacolumn)
                    else:
                        sumcost = 0
                        # numsubsets = int(math.log(len(values)/50)) + 1
                        numsubsets = 1
                        for s in range(numsubsets):
                            np.random.shuffle(values)
                            # NEW CODE
                            if len(values) > len(datacolumn):
                                np.random.shuffle(values)
                                cost = self.computeCost(values[:min(len(datacolumn), 50)], datacolumn)
                                cost = cost * 50 / len(datacolumn)
                            else:
                                cost = self.computeCost(values[:50], datacolumn)
                            sumcost += cost
                        avgcost = sumcost / numsubsets
                        cost = avgcost  # * len(values)/50

                    # if cost > thres: continue

                except TimeoutError:
                    return None
                # update top k
                # new code
                if cost < topK[-1][0]:
                    topK = self.update_topk(topK, dict_key, cost)

        # no matched columns within the given threshold. reject this column
        # if len(topK) == 0:
        #    return None
        return topK[0][1]
        predictions = []
        # for x in topK:
        #     predictions.append(x)
        for x in topK:
            # COMMENT
            result = self.resolveTriple(x[1])
            # COMMENT END
            # result = x[1]
            if result == None:
                continue
            elif len(result) < 3:
                continue

            predictions.append(list((result, x[0])))

        # if a pattern is detected, add label to return list
        if isPattern:
            if len(predictions) != 0:
                predictions[-1] = ('t', l, None)
            else:
                predictions.append(('t', l, None))

        return predictions

    def predict(self, eType, values, distributions, dtype='int'):
        # retrieve labels by mapping function
        # set the threshold for min-cost
        thres = abs(0.3 * sum(values))
        isPattern = False
        # check pre-defined patterns
        if dtype == 'int':
            (isPattern, l) = self.patternMatching(values)

        # TODO: REMOVE PATTERN MATCHING FOR TESTING PURPOSE
        isPattern = False

        allcolumns = distributions.items()

        # pruning
        queue, topK = self.createOrder(values, allcolumns)
        numCols = len(queue)

        pruned1, pruned2 = 0, 0
        while len(queue) != 0:
            data = heapq.heappop(queue)
            # new code
            datacolumn = distributions[data[1]]
            if len(datacolumn) > 5000:
                random.shuffle(datacolumn)
                datacolumn = datacolumn[:5000]
            currentMax = topK[-1][0]
            # new code
            datacolumn = self.remove_outliers(datacolumn)
            est = self.pruneUp(values, datacolumn, topK)
            if est > currentMax or est == -1:
                # new code
                pruned1 += 1

            elif self.pruneDown(values, datacolumn, topK) > currentMax:
                # new code
                pruned2 += 1

            else:
                try:
                    if len(values) < 51:
                        # NEW CODE
                        if len(values) > len(datacolumn):
                            np.random.shuffle(values)
                            # number_of_rows_to_check = max(len(datacolumn), min(len(datacolumn), 10))
                            number_of_rows_to_check = len(datacolumn)
                            cost = self.computeCost(values[:number_of_rows_to_check], datacolumn)
                            cost = cost * len(values) / number_of_rows_to_check
                        else:
                            cost = self.computeCost(values, datacolumn)

                        cost = cost / len(values)
                    else:
                        sumcost = 0
                        # numsubsets = int(math.log(len(values)/50)) + 1
                        numsubsets = 1
                        for s in range(numsubsets):
                            np.random.shuffle(values)
                            # NEW CODE
                            if len(values) > len(datacolumn):
                                # number_of_rows_to_check = min(max(len(datacolumn), min(len(datacolumn), 10)), 50)
                                number_of_rows_to_check = min(len(datacolumn), 50)
                                np.random.shuffle(values)
                                cost = self.computeCost(values[:number_of_rows_to_check], datacolumn)
                                cost = cost * 50 / number_of_rows_to_check
                            else:
                                cost = self.computeCost(values[:50], datacolumn)
                            sumcost += cost
                        avgcost = sumcost / numsubsets
                        cost = avgcost  # * len(values)/50

                        cost = cost / 50

                    # if cost > thres: continue

                except TimeoutError:
                    return None
                # update top k
                # NEW CODE 16 OCT
                # cost = cost / (sum(values) / len(values))
                if cost < topK[-1][0]:
                    topK = self.update_topk(topK, data[1], cost)

        # no matched columns within the given threshold. reject this column
        # if len(topK) == 0:
        #    return None

        predictions = []
        # for x in topK:
        #     predictions.append(x)
        for x in topK:
            # COMMENT
            result = self.resolveTriple(x[1])
            # COMMENT END
            # result = x[1]
            if result == None:
                continue
            elif len(result) < 3:
                continue

            predictions.append(list((result, x[0])))

        # if a pattern is detected, add label to return list
        if isPattern:
            if len(predictions) != 0:
                predictions[-1] = ('t', l, None)
            else:
                predictions.append(('t', l, None))

        return predictions

    def ksdistance(self, eType, values, distributions, dtype='int'):

        allcolumns = distributions.items()
        scores = []

        for c in allcolumns:
            if len(c[1]) == 0:
                scores.append([c[0], -1])
                continue
            statistics, pval = ks_2samp(values, c[1])
            scores.append([c[0], pval])

        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:min(self.k, len(scores))]
        predictions = []
        for x in scores:
            result = self.resolveTriple(x[0])
            if result == None:
                continue
            elif len(result) < 3:
                continue

            result = list(result)
            if result[2] == None:
                result[2] == ' '

            predictions.append(result)

        return predictions

    def find_in_dbpedia(self, eType):
        rv = {}
        query = '''SELECT * FROM distribution WHERE type LIKE ?; '''
        self.cursor.execute(query, ("%" + eType + "%",))
        row = self.cursor.fetchone()
        while row != None:
            if (row[0], row[1], row[2]) not in rv:
                rv[(row[0], row[1], row[2])] = []
            rv[(row[0], row[1], row[2])].append(row[3])
            row = self.cursor.fetchone()
        return rv

    def update_distribution_using_average_v1(self, distributions, values):
        with open("conversion_rates_v2.json", "r") as f:
            conversion_rates = json.load(f)
        with open("id_to_label_mapping.json", "r") as f:
            id_to_label_mapping = json.load(f)
        with open("label_to_id_mapping.json", "r") as f:
            label_to_id_mapping = json.load(f)
        avg_values = sum(values) / len(values)
        properties_dict = {}
        final_distribution = {}
        for key in distributions:
            # INJA BIA
            distributions[key] = self.remove_outliers(distributions[key])
            if len(distributions[key]) < 3:
                continue
            avg = sum(distributions[key]) / len(distributions[key])
            if key[1] not in properties_dict:
                properties_dict[key[1]] = []
            properties_dict[key[1]].append((key, avg))

            if key[2] not in id_to_label_mapping:
                continue
            unit = id_to_label_mapping[key[2]]
            if unit not in conversion_rates:
                continue
            for new_unit in conversion_rates[unit]:
                new_avg = avg * conversion_rates[unit][new_unit]
                new_unit_id = label_to_id_mapping[new_unit] if new_unit in label_to_id_mapping else 1
                properties_dict[key[1]].append(
                    ((key[0], key[1], new_unit_id), new_avg, key, conversion_rates[unit][new_unit]))

        for key in properties_dict:
            min_dif = abs(properties_dict[key][0][1] - avg_values)
            min_tpu = properties_dict[key][0]
            for tpu in properties_dict[key]:
                if abs(tpu[1] - avg_values) < min_dif:
                    min_dif = abs(tpu[1] - avg_values)
                    min_tpu = tpu
            if min_tpu[0] in distributions:
                final_distribution[min_tpu[0]] = distributions[min_tpu[0]]
            else:
                final_distribution[min_tpu[0]] = [i * min_tpu[3] for i in distributions[min_tpu[2]]]
        return final_distribution

    def update_distribution_using_median_v1(self, distributions, values):
        with open("conversion_rates_v2.json", "r") as f:
            conversion_rates = json.load(f)
        with open("id_to_label_mapping.json", "r") as f:
            id_to_label_mapping = json.load(f)
        with open("label_to_id_mapping.json", "r") as f:
            label_to_id_mapping = json.load(f)
        mean_values = values[int(len(values) / 2)]
        properties_dict = {}
        final_distribution = {}
        for key in distributions:
            # INJA BIA
            distributions[key] = self.remove_outliers(distributions[key])
            if len(distributions[key]) < 3:
                continue
            mean = distributions[key][int(len(distributions[key]) / 2)]
            if key[1] not in properties_dict:
                properties_dict[key[1]] = []
            properties_dict[key[1]].append((key, mean))

            if key[2] not in id_to_label_mapping:
                continue
            unit = id_to_label_mapping[key[2]]
            if unit not in conversion_rates:
                continue
            for new_unit in conversion_rates[unit]:
                new_mean = mean * conversion_rates[unit][new_unit]
                new_unit_id = label_to_id_mapping[new_unit] if new_unit in label_to_id_mapping else 1
                properties_dict[key[1]].append(
                    ((key[0], key[1], new_unit_id), new_mean, key, conversion_rates[unit][new_unit]))

        for key in properties_dict:
            min_dif = abs(properties_dict[key][0][1] - mean_values)
            min_tpu = properties_dict[key][0]
            for tpu in properties_dict[key]:
                if abs(tpu[1] - mean_values) < min_dif:
                    min_dif = abs(tpu[1] - mean_values)
                    min_tpu = tpu
            if min_tpu[0] in distributions:
                final_distribution[min_tpu[0]] = distributions[min_tpu[0]]
            else:
                final_distribution[min_tpu[0]] = [i * min_tpu[3] for i in distributions[min_tpu[2]]]
        return final_distribution

    def get_average_of_second_and_third_quarter(self, data):
        num = 0
        sum = 0
        for i in range(int(len(data) / 4), 3 * int(len(data) / 4)):
            num += 1
            sum += data[i]
        return sum / num

    def number_of_common_member(self, a, b):
        result = [i for i in a if i in b]
        return len(result)

    def add_new_data_to_column(self, distributions, key, id_to_label_mapping, conversion_rates):
        column = distributions[key]
        if len(column) == 0:
            return column
        if key[2] not in id_to_label_mapping:
            return column
        unit = id_to_label_mapping[key[2]]
        if unit not in conversion_rates:
            return column
        for key2 in distributions:
            if len(distributions[key2]) == 0 or key2[0] != key[0] or key2[1] != key[1] or key2[2] == key[2]:
                continue
            if key2[2] not in id_to_label_mapping:
                continue

            unit2 = id_to_label_mapping[key2[2]]
            if unit not in conversion_rates[unit2]:
                continue
            conversion_rate = conversion_rates[unit2][unit]
            new_data = [i * conversion_rate for i in distributions[key2]]
            common_members = self.number_of_common_member(column, new_data)
            coverage = common_members / min(len(column), len(new_data))
            if coverage < 0.5:
                column.extend(new_data)
        return column

    def update_distribution_using_average_of_second_and_third_quarter_v1(self, distributions, values):
        if len(values) < 4:
            return self.update_distribution_using_median_v1(distributions, values)
        with open("conversion_rates_v2.json", "r") as f:
            conversion_rates = json.load(f)
        with open("id_to_label_mapping.json", "r") as f:
            id_to_label_mapping = json.load(f)
        with open("label_to_id_mapping.json", "r") as f:
            label_to_id_mapping = json.load(f)
        avg_values = self.get_average_of_second_and_third_quarter(values)
        properties_dict = {}
        final_distribution = {}
        for key in distributions:
            distributions[key] = self.remove_outliers(distributions[key])
            if len(distributions[key]) < 4:
                continue
            # column_values = self.add_new_data_to_column(distributions, key, id_to_label_mapping, conversion_rates)
            column_values = distributions[key]
            avg = self.get_average_of_second_and_third_quarter(column_values)
            if key[1] not in properties_dict:
                properties_dict[key[1]] = []
            properties_dict[key[1]].append((key, avg))

            if key[2] not in id_to_label_mapping:
                continue
            unit = id_to_label_mapping[key[2]]
            if unit not in conversion_rates:
                continue
            for new_unit in conversion_rates[unit]:
                new_avg = avg * conversion_rates[unit][new_unit]
                new_unit_id = label_to_id_mapping[new_unit] if new_unit in label_to_id_mapping and type(label_to_id_mapping[new_unit]) == str else '1'
                properties_dict[key[1]].append(
                    ((key[0], key[1], new_unit_id), new_avg, key, conversion_rates[unit][new_unit]))

        for key in properties_dict:
            min_dif = [np.inf, np.inf, np.inf]
            min_tpu = [(None, np.inf), (None, np.inf), (None, np.inf)]
            for tpu in properties_dict[key]:
                distance = abs(tpu[1] - avg_values) / max(abs(tpu[1]), abs(avg_values), 1)  # * (max(tpu[1], avg_values) / min(tpu[1], avg_values))
                is_key_duplicate = False
                for v in min_tpu:
                    if v[0] == None:
                        continue
                    if v[0][0] == tpu[0]:
                        is_key_duplicate = True
                        if distance < v[1]:
                            min_dif.remove(v[1])
                            min_dif.append(distance)
                            min_tpu.remove(v)
                            min_tpu.append((tpu, distance))
                            break
                if is_key_duplicate:
                    continue
                if distance < max(min_dif):
                    for v in min_tpu:
                        if v[1] == max(min_dif):
                            min_tpu.remove(v)
                            break
                    min_dif.remove(max(min_dif))
                    min_dif.append(distance)  # * (max(tpu[1], avg_values) / min(tpu[1], avg_values))
                    min_tpu.append((tpu, distance))
            for tpu in min_tpu:
                if tpu[1] == np.inf:
                    continue
                tpu = tpu[0]
                if tpu[0] in distributions:
                    final_distribution[tpu[0]] = distributions[tpu[0]]
                else:
                    final_distribution[tpu[0]] = [i * tpu[3] for i in distributions[tpu[2]]]
        return final_distribution

    def update_distribution_using_SAND(self, distributions, values, dtype):
        with open("conversion_rates_v2.json", "r") as f:
            conversion_rates = json.load(f)
        with open("id_to_label_mapping.json", "r") as f:
            id_to_label_mapping = json.load(f)
        with open("label_to_id_mapping.json", "r") as f:
            label_to_id_mapping = json.load(f)
        new_dict = {}
        for key in distributions.keys():
            distributions[key] = self.remove_outliers(distributions[key])
            if len(distributions[key]) < 4:
                continue
            if key[1] not in new_dict:
                new_dict[key[1]] = {}
            new_dict[key[1]][key] = distributions[key]

            if key[2] not in id_to_label_mapping:
                continue
            unit = id_to_label_mapping[key[2]]
            if unit not in conversion_rates:
                continue
            for new_unit in conversion_rates[unit]:
                cr = conversion_rates[unit][new_unit]
                new_unit_id = label_to_id_mapping[new_unit] if new_unit in label_to_id_mapping else 1
                if (key[0], key[1], new_unit_id) not in new_dict[key[1]]:
                    new_dict[key[1]][(key[0], key[1], new_unit_id)] = [i * cr for i in distributions[key]]
                else:
                    new_dict[key[1]][(key[0], key[1], new_unit_id)].extend([i * cr for i in distributions[key]])
        predicted = []
        for dic in new_dict:
            p = self.predict_without_order('', values, new_dict[dic], dtype)
            if p:
                predicted.append((p, new_dict[dic][p]))
        return predicted

    def update_distribution_using_average_v2(self, distributions, values):
        avg_values = sum(values) / len(values)
        properties_dict = {}
        final_distribution = {}
        for key in distributions:
            if len(distributions[key]) < 3:
                continue
            avg = sum(distributions[key]) / len(distributions[key])
            if key[1] not in properties_dict:
                properties_dict[key[1]] = []
            properties_dict[key[1]].append((key, avg))
        for key in properties_dict:
            min_dif = abs(properties_dict[key][0][1] - avg_values)
            min_tpu = properties_dict[key][0]
            for tpu in properties_dict[key]:
                if abs(tpu[1] - avg_values) < min_dif:
                    min_dif = abs(tpu[1] - avg_values)
                    min_tpu = tpu
            final_distribution[min_tpu[0]] = distributions[min_tpu[0]]
        return final_distribution

    def update_distribution_using_ks_test_v2(self, distributions, values):
        properties_dict = {}
        final_distribution = {}
        for key in distributions:
            if len(distributions[key]) == 0:
                continue
            similarity, _ = ks_2samp(distributions[key], values)
            if key[1] not in properties_dict and similarity:
                properties_dict[key[1]] = []
            if similarity:
                properties_dict[key[1]].append((key, similarity))
        print(properties_dict)
        for key in properties_dict:
            max_similarity = properties_dict[key][0][1]
            max_tpu = properties_dict[key][0][0]
            for tpu in properties_dict[key]:
                if tpu[1] > max_similarity:
                    max_similarity = tpu[1]
                    max_tpu = tpu[0]
            final_distribution[max_tpu] = distributions[max_tpu]
        return final_distribution

    def divide_column(self, values):
        values = self.remove_outliers(values)
        values.sort()
        new_data = [[]]
        for d in values:
            if len(new_data[-1]) > 0:
                if new_data[-1][-1] * 3 < d:
                    new_data.append([])
            new_data[-1].append(d)
            new_data[-1].sort()
        list_of_new_columns = []
        for d in new_data:
            if len(d) < 4:
                continue
            new_d = [x for x in d]
            list_of_new_columns.append(new_d)
        return list_of_new_columns

    def divide_column_using_std_dev(self, values):
        values = self.remove_outliers(values)
        values.sort()
        new_data = [[]]
        for v in values:
            if len(new_data[-1]) < 3:
                new_data[-1].append(v)
                continue
            test_data = []
            test_data.extend(new_data[-1])
            test_data.append(v)
            std_dev = statistics.stdev(test_data)
            mean = sum(test_data) / len(test_data)
            if std_dev == 0 or (abs(v - mean) / std_dev) > 3 or 3 * new_data[-1][-1] < v:
                new_data[-1] = self.remove_outliers(new_data[-1])
                new_data.append([v])
            else:
                new_data[-1].append(v)
        new_data[-1] = self.remove_outliers(new_data[-1])
        list_of_new_columns = []
        for d in new_data:
            if len(d) < 4:
                continue
            new_d = [x for x in d]
            list_of_new_columns.append(new_d)
        return list_of_new_columns

    def update_predictions_old_method(self, predictions):
        new_predictions = []
        for prediction in predictions:
            new_predictions.extend(prediction)
        new_predictions.sort(key=lambda x: x[1])
        return new_predictions[:self.k]

    def check_property_presence_in_prediction(self, prop, predictions):
        for pred in predictions:
            if prop == pred[0][1]:
                return True
        return False

    def update_predictions(self, predictions):
        new_predictions = []
        for i in range(len(predictions)):
            for pred in predictions[i]:
                if self.check_property_presence_in_prediction(pred[0][1], new_predictions):
                    continue
                cost = pred[1]
                for j in range(len(predictions)):
                    if i == j:
                        continue
                    max_cost = 0
                    for pred2 in predictions[j]:
                        if pred2[0][1] == pred[0][1]:
                            max_cost = pred2[1]
                            break
                        if max_cost < pred2[1]:
                            max_cost = pred2[1]
                    cost += max_cost
                pred[1] = cost
                new_predictions.append(pred)

        new_predictions.sort(key=lambda x: x[1])
        print(len(new_predictions))
        return new_predictions[:self.k]

    def predict_divided_columns(self, eType, list_of_columns, distributions, dtype):
        self.k = self.k + 5
        predictions = []
        max_cost = 0
        for values in list_of_columns:
            prediction = self.predict(eType, values, distributions, dtype)
            for pred in prediction:
                if pred[1] > max_cost:
                    max_cost = pred[1]
            predictions.append(prediction)
        self.k = self.k - 5
        return self.update_predictions_old_method(predictions)

    def divide_and_predict_column(self, eType, values, distributions, dtype):
        list_of_columns = self.divide_column_using_std_dev(values)
        return self.predict_divided_columns(eType, list_of_columns, distributions, dtype)

    def common_predictions_for_two_columns_without_common_entity_property(self, c1_predictions, c2_predictions):
        final_predictions = []
        for prediction in c1_predictions:
            for prediction2 in c2_predictions:
                if prediction[0][1] == prediction2[0][1]:
                    new_pred = True
                    for final_pred in reversed(final_predictions):
                        if final_pred[0][1] == prediction[0][1]:
                            if final_pred[1] > prediction[1]:
                                final_predictions.remove(final_pred)
                            else:
                                new_pred = False
                    if new_pred:
                        final_predictions.append([(prediction[0][0], prediction[0][1], (prediction[0][2], prediction2[0][2])), prediction[1] + prediction2[1]])
        final_predictions.sort(key=lambda x: x[-1])
        return final_predictions if len(final_predictions) <= self.k else final_predictions[:self.k]

    def common_predictions_for_two_columns(self, c1_predictions, c2_predictions):
        final_predictions = []
        for prediction in c1_predictions:
            for prediction2 in c2_predictions:
                if prediction[0][1] == prediction2[0][1]:

                    final_predictions.append([(prediction[0][0], prediction[0][1], (prediction[0][2], prediction2[0][2])), prediction[1] + prediction2[1]])
        final_predictions.sort(key=lambda x: x[-1])
        return final_predictions if len(final_predictions) <= self.k else final_predictions[:self.k]

    def common_predictions(self, predictions):
        final_predictions = self.common_predictions_for_two_columns(predictions[0], predictions[1])
        for i in range(2, len(predictions)):
            final_predictions = self.common_predictions_for_two_columns(final_predictions, predictions[i])
        return final_predictions

    def common_predictions_without_common_entity_property(self, predictions):
        final_predictions = self.common_predictions_for_two_columns_without_common_entity_property(predictions[0], predictions[1])
        for i in range(2, len(predictions)):
            final_predictions = self.common_predictions_for_two_columns_without_common_entity_property(final_predictions, predictions[i])
        return final_predictions

    def get_conversion_rate(self, units):
        print(units)
        with open("conversion_rates_v2.json", "r") as f:
            conversion_rates = json.load(f)
        if units[0] not in conversion_rates:
            return -2
        return conversion_rates[units[0]][units[1]] if units[1] in conversion_rates[units[0]] else -3

    def predict_all_mixed_unit_column(self, values, distributions, eType, dtype):
        if len(values) < 6:
            return {}
        final_predictions_for_units = {}
        for number_of_clusters in range(2, 7):
            final_predictions_for_units[number_of_clusters] = []
            clusters = normal_k_means_clustering(values, number_of_clusters)
            # clusters = k_means(values, number_of_clusters)
            final_predictions = []
            self.k += 10
            for key in clusters:
                cluster_values = self.remove_outliers(clusters[key])
                if len(cluster_values) == 0:
                    continue
                cluster_distributions = self.update_distribution_using_average_of_second_and_third_quarter_v1(
                    distributions, cluster_values)
                cluster_predictions = self.predict(eType, cluster_values, cluster_distributions, dtype)
                final_predictions.append(cluster_predictions)
            self.k -= 10
            if len(final_predictions) >= 2:
                final_predictions = self.common_predictions_without_common_entity_property(final_predictions)
            else:
                continue
            final_predictions_for_units[number_of_clusters] = final_predictions
        return final_predictions_for_units

    def predict_mixed_unit_column(self, values, distributions, eType, dtype):
        if len(values) < 6:
            return []
        number_of_clusters = find_number_of_clusters(values)
        print("number_of_clusters", number_of_clusters)
        clusters = k_means(values, number_of_clusters)
        for key in clusters:
            print(clusters[key])
        final_predictions = []
        self.k += 10
        for key in clusters:
            cluster_values = self.remove_outliers(clusters[key])
            if len(cluster_values) == 0:
                self.k -= 10
                return []
            cluster_distributions = self.update_distribution_using_average_of_second_and_third_quarter_v1(
                distributions, cluster_values)
            cluster_predictions = self.predict(eType, cluster_values, cluster_distributions, dtype)
            final_predictions.append(cluster_predictions)
        self.k -= 10
        if len(final_predictions) >= 2:
            final_predictions = self.common_predictions_without_common_entity_property(final_predictions)
        else:
            return []
        return final_predictions

    def predict_two_different_columns(self, values, distributions, eType, dtype, semantic):
        c1_values, c2_values = k_means(values, 2)
        print("clustering finished")
        c1_values = self.remove_outliers(c1_values)
        c2_values = self.remove_outliers(c2_values)
        if len(c1_values) == 0 or len(c2_values) == 0:
            return []
        distributions_c1 = self.update_distribution_using_average_of_second_and_third_quarter_v1(distributions,
                                                                                                 c1_values)
        self.k += 5
        c1_predictions = self.predict(eType, c1_values, distributions_c1, dtype)
        distributions_c2 = self.update_distribution_using_average_of_second_and_third_quarter_v1(distributions,
                                                                                                 c2_values)
        c2_predictions = self.predict(eType, c2_values, distributions_c2, dtype)
        self.k -= 5
        # predictions = self.divide_and_predict_column(eType, values, distributions, dtype)
        final_predictions = self.common_predictions_without_common_entity_property(c1_predictions, c2_predictions)
        print("****")
        print("semantic: ", semantic)
        print("c1 prediction: ", c1_predictions)
        print("c2 prediction: ", c2_predictions)
        print("final predictions: ", final_predictions)
        return final_predictions

    def check_mixed_unit_and_get_final_prediction(self, final_predictions, unit_numbers=0):
        if not final_predictions:
            return []
        cost_for_units = {}
        for number_of_units in final_predictions:
            cost_for_units[number_of_units] = 0
            for pred in final_predictions[number_of_units]:
                cost_for_units[number_of_units] += pred[1]
            cost_for_units[number_of_units] = cost_for_units[number_of_units] / len(final_predictions[number_of_units]) if len(final_predictions[number_of_units]) > 0 else -1
        min_cost = -1
        number_of_units = 1
        for unit in cost_for_units:
            if cost_for_units[unit] == -1:
                continue
            if cost_for_units[unit] < min_cost or min_cost == -1:
                min_cost = cost_for_units[unit]
                number_of_units = unit
        print("number of units: ", number_of_units)
        if number_of_units == unit_numbers:
            NUM_CLUSTERS_K_MEANS[number_of_units] += 1
        return final_predictions[number_of_units]

    def check_if_it_is_mixed_unit_and_get_final_prediction(self, final_predictions_for_not_mixed_unit, final_predictions_for_mixed_unit):
        cost_for_mixed_unit = 0
        cost_for_not_mixed_unit = 0
        for pred in final_predictions_for_mixed_unit:
            cost_for_mixed_unit += pred[1]
        for pred in final_predictions_for_not_mixed_unit:
            cost_for_not_mixed_unit += pred[1]
        cost_for_mixed_unit = cost_for_mixed_unit / len(final_predictions_for_mixed_unit) if len(
            final_predictions_for_mixed_unit) > 0 else -1
        cost_for_not_mixed_unit = cost_for_not_mixed_unit / len(final_predictions_for_not_mixed_unit) if len(
            final_predictions_for_not_mixed_unit) > 0 else -1
        if cost_for_mixed_unit != -1 and (cost_for_not_mixed_unit == -1 or cost_for_mixed_unit < cost_for_not_mixed_unit):
            self.number_of_mixed_units_columns += 1
            print("It's a mixed unit")
            return final_predictions_for_mixed_unit
        self.number_of_not_mixed_units_columns += 1
        print("It's not a mixed unit")
        return final_predictions_for_not_mixed_unit

    def update_query_column_size(self, values, query_column_size):
        np.random.shuffle(values)
        values = values[:query_column_size]
        return values

    def annotate(self, query_column_size=None):
        self.connect()
        signal.signal(signal.SIGALRM, TimeoutHandler)

        total = 0
        correctUnit = 0
        correctLabel = 0
        correctLabel1 = 0
        correctLabel3 = 0
        unable = 0

        # outfile = open('WTResult.txt','w')
        short = open('result_short.txt', 'w')
        # ent = open('typeDetectWDC.txt', 'w')
        eTypePred = 0
        easy_dataset = "dataset/new_easy_dataset.txt"
        with open(easy_dataset, 'r') as tbf:
            i = -1
            # for i in range(110): line = tbf.readline().strip()
            line = tbf.readline().strip()
            x = 0
            try:
                while line != '':
                    print("k = 1: ", correctLabel1, correctUnit, total)
                    print("k = 3: ", correctLabel3, correctUnit, total)
                    print("k = 5: ", correctLabel, correctUnit, total)
                    i += 1
                    table = json.loads(line)

                    headers = table['header']
                    values = table['values']
                    entities = table['entity']
                    unit_numbers = table['unit_numbers'] if "unit_numbers" in table else 0

                    unit = table['unit']
                    semantic = table['semantic'].lower()
                    # year is usually not in kb
                    if semantic == 'year':
                        line = tbf.readline().strip()
                        continue
                    eType = table['eType']

                    try:
                        eType = self.resolve(table['eType'], reverse=True)
                        if len(eType) == 0:
                            line = tbf.readline().strip()
                            continue
                    except:
                        eType = self.getEntityType(entities)
                        # cannot find matched type
                        if eType == None:
                            line = tbf.readline().strip()
                            continue

                        types = [self.resolve(t) for t in eType]
                        # ent.write("header: " + headers[0] + '\n sample entity: ' + entities[0] + '\n')
                        # ent.write("predicted type: " + str(types) + '\n--------------\n\n')
                        # ent.flush()
                        # line = tbf.readline().strip()
                        # continue
                    '''
                    eType = self.getEntityType(entities)
                    # cannot find matched type
                    if eType == None:
                        line = tbf.readline().strip()
                        continue
                    '''

                    ### test entity type prediction
                    '''
                    eType = table['eType'].lower()
                    predType = self.getEntityType(entities)
                    # cannot find matched type
                    if eType == None and predType == None:
                        line = tbf.readline().strip()
                        continue
                    total += 1
                    types = [self.resolve(t).lower() for t in predType]
                    if eType in types:
                        eTypePred += 1
                    print (eType, types)
                    print (eTypePred, total)
                    line = tbf.readline().strip()
                    continue
                    '''
                    # print(eType, values, i)
                    # eType = self.get_parents_to_the_root(hierarchy_graph, eType)
                    distributions = self.getDistribution(eType)
                    if len(distributions) == 0:
                        line = tbf.readline().strip()
                        continue
                    dtype = 'float'
                    # values = ';'.join(values).replace(',', '').split(';')
                    # NEW CODE
                    values = [v.replace(',', '') for v in values]
                    try:
                        values = list(map(int, values))
                        dtype = 'int'
                    except ValueError:
                        try:
                            values = list(map(float, values))
                        except ValueError:
                            t = []
                            for x in values:
                                try:
                                    t.append(float(x))
                                except ValueError:
                                    pass
                            if len(t) < len(values) / 2:
                                line = tbf.readline().strip()
                                continue
                            values = t
                    # the largest value is the 'total' of all other values
                    v = sorted(values, reverse=True)
                    if sum(values[1:]) == v[0]: values.remove(v[0])

                    if query_column_size:
                        values = self.update_query_column_size(values, query_column_size)

                    # check kardane sand va avg
                    # d2 = self.update_distribution_using_average_of_second_and_third_quarter_v1(distributions, values)
                    # distributions = self.update_distribution_using_SAND(distributions, values, dtype)
                    # print(len(distributions))
                    # for key in distributions:
                    #     if key[0] not in d2:
                    #         print("*** SAND")
                    #         print(key[0])
                    #         for d in d2:
                    #             if d[1] == key[0][1]:
                    #                 print("Avg")
                    #                 print(d)
                    #                 print("distribution e avg")
                    #                 print(d2[d])
                    #                 print(len(d2[d]))
                    #         print("distribution e sand")
                    #         print(key[1])
                    #         print(len(key[1]))
                    #         print(values)
                    # line = tbf.readline().strip()
                    # continue

                    # TWO PREDICTIONS
                    if not query_column_size:
                        final_predictions_for_mixed_unit = self.predict_all_mixed_unit_column(values, distributions, eType, dtype)
                        distributions = self.update_distribution_using_average_of_second_and_third_quarter_v1(distributions, values)
                        final_predictions_for_mixed_unit[1] = self.predict(eType, values, distributions, dtype)
                        final_predictions = self.check_mixed_unit_and_get_final_prediction(final_predictions_for_mixed_unit, unit_numbers=unit_numbers)
                    else:
                        distributions = self.update_distribution_using_average_of_second_and_third_quarter_v1(
                            distributions, values)
                        final_predictions = self.predict(eType, values, distributions, dtype)
                    print("semantic: ", semantic)
                    print("final predictions: ", final_predictions)
                    # line = tbf.readline().strip()
                    # continue
                    # predictions = self.ksdistance(eType, values, distributions, dtype)

                    if final_predictions is None:
                        unable += 1
                        line = tbf.readline().strip()
                        continue
                    elif len(final_predictions) == 0:
                        unable += 1
                        line = tbf.readline().strip()
                        continue

                    # # NEW CODE
                    # for prediction in predictions:
                    #     p_semantic = [x[0][1] for x in prediction]
                    #     p_unit = [x[0][2] for x in prediction]
                    p_semantic = [x[0][1] for x in final_predictions]
                    p_unit = [x[0][2] for x in final_predictions]

                    # check correctness
                    total += 1
                    correct_sem = ""
                    index = 0
                    predicted_unit = None
                    for p_sem in p_semantic:
                        index += 1
                        if p_sem and (semantic in p_sem or p_sem in semantic):
                            predicted_unit = p_unit[index - 1] if isinstance(p_unit[index - 1], list) else [p_unit[index - 1]]
                            if index == 1:
                                correctLabel1 += 1
                            if index <= 3:
                                correctLabel3 += 1
                            correctLabel += 1
                            correct_sem = p_sem
                            break
                        elif p_sem and 'number of' in p_sem and 'number of' in semantic:
                            wrong = False
                            predicted_unit = p_unit[index - 1] if isinstance(p_unit[index - 1], list) else [p_unit[index - 1]]
                            correctLabel += 1
                            if index == 1:
                                correctLabel1 += 1
                            if index <= 3:
                                correctLabel3 += 1
                            correct_sem = p_sem
                            break
                    # TWO PREDICTIONS
                    # if correct_sem:
                    #     for pred in final_predictions:
                    #         if pred[0][1] == correct_sem:
                    #             correct_units = pred[0][2]
                    #             break
                    #     print("conversion rate: ", self.get_conversion_rate(correct_units))
                    #     print("correct conversion rate: ", statistics.mean(c1_values) / statistics.mean(c2_values) if statistics.mean(c2_values) != 0 else 0)
                    #     print("correct conversion rate2: ", statistics.mean(c2_values) / statistics.mean(c1_values) if statistics.mean(c1_values) != 0 else 0)

                    if predicted_unit and unit in predicted_unit: correctUnit += 1

                    print(i, headers, unit, file=short)
                    print('labels given by mapping:', file=short)
                    for x in final_predictions: print(x, file=short)
                    print('\n -------------------------', file=short)

                    # short.flush()

                    line = tbf.readline().strip()
                print('k=', self.k, correctLabel, correctUnit, total)
                print('k=', 3, correctLabel3, correctUnit, total)
                print('k=', 1, correctLabel1, correctUnit, total)

            except EOFError:
                pass

        # outfile.close()
        short.close()
        # ent.close()

        self.clean()

    def prepare_values(self, values):
        try:
            values = list(map(int, values))
            dtype = 'int'
        except ValueError:
            try:
                values = list(map(float, values))
            except ValueError:
                t = []
                for x in values:
                    try:
                        t.append(float(x))
                    except ValueError:
                        pass
                if len(t) < len(values) / 2:
                    return None
                values = t
        t = []
        for value in values:
            if not np.isnan(value):
                t.append(float(value))
        if len(t) < len(values) / 2:
            return None
        values = t
        return values

    def predict_two_separated_unit_column(self, values, distributions, eType, dtype):
        if len(values[0]) < 3 or len(values[1]) < 3:
            print("number of values are less than 3")
            return {}
        final_predictions_for_units = {}
        final_predictions_for_units[2] = []
        final_predictions = []
        self.k += 10
        for value in values:
            cluster_values = self.remove_outliers(value)
            if len(cluster_values) == 0:
                print("cluster values after removing outliers are none")
                continue
            cluster_distributions = self.update_distribution_using_average_of_second_and_third_quarter_v1(
                distributions, cluster_values)
            cluster_predictions = self.predict(eType, cluster_values, cluster_distributions, dtype)
            final_predictions.append(cluster_predictions)
        self.k -= 10
        if len(final_predictions) >= 2:
            final_predictions = self.common_predictions_without_common_entity_property(final_predictions)
        elif len(final_predictions) == 1:
            final_predictions = final_predictions[0]
        else:
            print("number of predictions are less than 2")
            return {}
        final_predictions_for_units[2] = final_predictions
        return final_predictions_for_units

    def annotate_separated_units(self):
        self.connect()
        signal.signal(signal.SIGALRM, TimeoutHandler)

        total = 0
        correctUnit = 0
        correctLabel = 0
        correctLabel1 = 0
        correctLabel3 = 0
        unable = 0

        # outfile = open('WTResult.txt','w')
        short = open('result_short.txt', 'w')
        separated_units_dataset = "dataset/separated_units.txt"
        with open(separated_units_dataset, 'r') as tbf:
            i = -1
            # for i in range(110): line = tbf.readline().strip()
            line = tbf.readline().strip()
            x = 0
            try:
                while line != '':
                    print("k = 1: ", correctLabel1, correctUnit, total)
                    print("k = 3: ", correctLabel3, correctUnit, total)
                    print("k = 5: ", correctLabel, correctUnit, total)
                    i += 1
                    table = json.loads(line)

                    headers = table['header']
                    values = table['values']
                    entities = table['entity']
                    unit_numbers = table['unit_numbers'] if "unit_numbers" in table else 0

                    unit = table['unit']
                    semantic = table['semantic'].lower()
                    # year is usually not in kb
                    if semantic == 'year':
                        line = tbf.readline().strip()
                        continue
                    eType = table['eType']

                    try:
                        eType = self.resolve(table['eType'], reverse=True)
                        if len(eType) == 0:
                            print("etype is none")
                            line = tbf.readline().strip()
                            continue
                    except:
                        eType = self.getEntityType(entities)
                        # cannot find matched type
                        if eType == None:
                            print("etype is none")
                            line = tbf.readline().strip()
                            continue

                        types = [self.resolve(t) for t in eType]
                        # ent.write("header: " + headers[0] + '\n sample entity: ' + entities[0] + '\n')
                        # ent.write("predicted type: " + str(types) + '\n--------------\n\n')
                        # ent.flush()
                        # line = tbf.readline().strip()
                        # continue
                    '''
                    eType = self.getEntityType(entities)
                    # cannot find matched type
                    if eType == None:
                        line = tbf.readline().strip()
                        continue
                    '''

                    ### test entity type prediction
                    '''
                    eType = table['eType'].lower()
                    predType = self.getEntityType(entities)
                    # cannot find matched type
                    if eType == None and predType == None:
                        line = tbf.readline().strip()
                        continue
                    total += 1
                    types = [self.resolve(t).lower() for t in predType]
                    if eType in types:
                        eTypePred += 1
                    print (eType, types)
                    print (eTypePred, total)
                    line = tbf.readline().strip()
                    continue
                    '''
                    # print(eType, values, i)
                    # eType = self.get_parents_to_the_root(hierarchy_graph, eType)
                    distributions = self.getDistribution(eType)
                    if len(distributions) == 0:
                        line = tbf.readline().strip()
                        print("DISTRIBUTION IS NONE")
                        continue
                    dtype = 'float'
                    # values = ';'.join(values).replace(',', '').split(';')
                    # NEW CODE
                    values[0] = [v.replace(',', '') for v in values[0]]
                    values[0] = self.prepare_values(values[0])
                    values[1] = [v.replace(',', '') for v in values[1]]
                    values[1] = self.prepare_values(values[1])
                    if values[0] is None or values[1] is None:
                        print("VALUES ARE NONE")
                        print(table)
                        print("---")
                        line = tbf.readline().strip()
                        continue


                    # check kardane sand va avg
                    # d2 = self.update_distribution_using_average_of_second_and_third_quarter_v1(distributions, values)
                    # distributions = self.update_distribution_using_SAND(distributions, values, dtype)
                    # print(len(distributions))
                    # for key in distributions:
                    #     if key[0] not in d2:
                    #         print("*** SAND")
                    #         print(key[0])
                    #         for d in d2:
                    #             if d[1] == key[0][1]:
                    #                 print("Avg")
                    #                 print(d)
                    #                 print("distribution e avg")
                    #                 print(d2[d])
                    #                 print(len(d2[d]))
                    #         print("distribution e sand")
                    #         print(key[1])
                    #         print(len(key[1]))
                    #         print(values)
                    # line = tbf.readline().strip()
                    # continue

                    # TWO PREDICTIONS
                    final_predictions_for_mixed_unit = self.predict_two_separated_unit_column(values, distributions,
                                                                                              eType,dtype)
                    # distributions = self.update_distribution_using_average_of_second_and_third_quarter_v1(
                    #     distributions, values[0] + values[1])
                    # final_predictions_for_mixed_unit[1] = self.predict(eType, values[0] + values[1], distributions, dtype)
                    try:
                        final_predictions = final_predictions_for_mixed_unit[2]
                    except:
                        print("final prediction is none")
                        print(table)
                        print("---")
                        line = tbf.readline().strip()
                        continue
                    print("semantic: ", semantic)
                    print("final predictions: ", final_predictions)
                    # line = tbf.readline().strip()
                    # continue
                    # predictions = self.ksdistance(eType, values, distributions, dtype)

                    if final_predictions is None:
                        unable += 1
                        line = tbf.readline().strip()
                        continue
                    elif len(final_predictions) == 0:
                        unable += 1
                        line = tbf.readline().strip()
                        continue

                    # # NEW CODE
                    # for prediction in predictions:
                    #     p_semantic = [x[0][1] for x in prediction]
                    #     p_unit = [x[0][2] for x in prediction]
                    p_semantic = [x[0][1] for x in final_predictions]
                    p_unit = [x[0][2] for x in final_predictions]

                    # check correctness
                    total += 1
                    correct_sem = ""
                    index = 0
                    predicted_unit = None
                    for p_sem in p_semantic:
                        index += 1
                        if p_sem and (semantic in p_sem or p_sem in semantic):
                            predicted_unit = p_unit[index - 1] if isinstance(p_unit[index - 1], list) else [p_unit[index - 1]]
                            if index == 1:
                                correctLabel1 += 1
                            if index <= 3:
                                correctLabel3 += 1
                            correctLabel += 1
                            correct_sem = p_sem
                            break
                        elif p_sem and 'number of' in p_sem and 'number of' in semantic:
                            wrong = False
                            predicted_unit = p_unit[index - 1] if isinstance(p_unit[index - 1], list) else [p_unit[index - 1]]
                            correctLabel += 1
                            if index == 1:
                                correctLabel1 += 1
                            if index <= 3:
                                correctLabel3 += 1
                            correct_sem = p_sem
                            break
                    # TWO PREDICTIONS
                    # if correct_sem:
                    #     for pred in final_predictions:
                    #         if pred[0][1] == correct_sem:
                    #             correct_units = pred[0][2]
                    #             break
                    #     print("conversion rate: ", self.get_conversion_rate(correct_units))
                    #     print("correct conversion rate: ", statistics.mean(c1_values) / statistics.mean(c2_values) if statistics.mean(c2_values) != 0 else 0)
                    #     print("correct conversion rate2: ", statistics.mean(c2_values) / statistics.mean(c1_values) if statistics.mean(c1_values) != 0 else 0)

                    if predicted_unit and unit in predicted_unit: correctUnit += 1

                    print(i, headers, unit, file=short)
                    print('labels given by mapping:', file=short)
                    for x in final_predictions: print(x, file=short)
                    print('\n -------------------------', file=short)

                    # short.flush()

                    line = tbf.readline().strip()
                print('k=', self.k, correctLabel, correctUnit, total)
                print('k=', 3, correctLabel3, correctUnit, total)
                print('k=', 1, correctLabel1, correctUnit, total)

            except EOFError:
                pass

        # outfile.close()
        short.close()
        # ent.close()

        self.clean()

    def read_values(self, file_name):
        values = []
        with open(file_name, 'r') as tbf:
            line = tbf.readline().strip()
            try:
                while line != '':
                    table = json.loads(line)
                    value = table['values']
                    values.append(value)
                    line = tbf.readline().strip()
            except EOFError:
                pass
        return values

    def annotate_dbpedia(self):
        self.connect()
        signal.signal(signal.SIGALRM, TimeoutHandler)

        total = 0
        correctUnit = 0
        correctLabel = 0
        unable = 0

        # outfile = open('WTResult.txt','w')
        short = open('result_short.txt', 'w')
        # ent = open('typeDetectWDC.txt', 'w')
        eTypePred = 0
        wikitable = "dataset/wiki_annotated_dbpedia4.txt"
        wdc = "dataset/wdc_dbpedia3.txt"
        tdv = "dataset/T2Dv2_v3.txt"
        tdv_v4 = "dataset/T2Dv2_v4.txt"
        tdv_v4_values = self.read_values(tdv_v4)
        with open(wdc, 'r') as tbf:
            i = -1
            # for i in range(110): line = tbf.readline().strip()
            line = tbf.readline().strip()
            skip_counter = 0
            tedad = 0
            try:
                while line != '':
                    print(correctLabel, correctUnit, total)
                    i += 1
                    table = json.loads(line)

                    headers = table['header']
                    values = table['values']
                    # if values in tdv_v4_values:
                    #     line = tbf.readline().strip()
                    #     skip_counter += 1
                    #     print("skip: ", str(skip_counter))
                    #     continue
                    entities = table['entity']

                    unit = table['unit']
                    semantic = table['semantic'].lower()
                    # year is usually not in kb
                    if semantic == 'year':
                        line = tbf.readline().strip()
                        continue
                    eType = table['eType']
                    # distributions = get_features_and_kinds_multi_thread(eType)
                    if len(distributions) == 0:
                        line = tbf.readline().strip()
                        continue
                    print(distributions)
                    break
                    dtype = 'float'
                    # values = ';'.join(values).replace(',', '').split(';')
                    # NEW CODE
                    values = [v.replace(',', '') for v in values]
                    try:
                        values = list(map(int, values))
                        dtype = 'int'
                    except ValueError:
                        try:
                            values = list(map(float, values))
                        except ValueError:
                            t = []
                            for x in values:
                                try:
                                    t.append(float(x))
                                except ValueError:
                                    pass
                            if len(t) < len(values) / 2:
                                line = tbf.readline().strip()
                                continue
                            values = t
                    # the largest value is the 'total' of all other values
                    v = sorted(values, reverse=True)
                    if sum(values[1:]) == v[0]: values.remove(v[0])

                    print("total: ", str(total))
                    predictions = self.predict(eType, values, distributions, dtype)
                    tedad += 1
                    print("semantic: ", semantic)
                    print("prediction: ", predictions)
                    # line = tbf.readline().strip()
                    # continue
                    # predictions = self.ksdistance(eType, values, distributions, dtype)

                    if predictions == None:
                        unable += 1
                        line = tbf.readline().strip()
                        continue
                    elif len(predictions) == 0:
                        unable += 1
                        line = tbf.readline().strip()
                        continue

                    p_semantic = [x[0][1].split('/')[-1] for x in predictions]
                    p_unit = [x[0][2] for x in predictions]

                    # check correctness
                    total += 1
                    for p_sem in p_semantic:
                        if p_sem and (semantic in p_sem or p_sem in semantic):
                            correctLabel += 1
                            break
                        elif p_sem and 'number of' in p_sem and 'number of' in semantic:
                            correctLabel += 1
                            break

                    if unit in p_unit: correctUnit += 1

                    print(i, headers, unit, file=short)
                    print('labels given by mapping:', file=short)
                    for x in predictions: print(x, file=short)
                    print('\n -------------------------', file=short)

                    # short.flush()

                    line = tbf.readline().strip()
                print('k=', self.k, correctLabel, correctUnit, total)

            except EOFError:
                pass

        # outfile.close()
        short.close()
        # ent.close()

        self.clean()

    def verify(self, semanticLabel, unit, labels):
        resolveQuery = '''select label from mapping where wid = ?;'''
        propertyMatch = 0
        unitMatch = 0

        for l in labels:
            prop = l[1]
            unit = l[2]

            try:
                self.cursor.execute(resolveQuery, (prop,))
                stringRepr = self.cursor.fetchone()[0]
                print(stringRepr, semanticLabel)
                if stringRepr.lower() == semanticLabel[1].lower():
                    propertyMatch = 1

                self.cursor.execute(resolveQuery, (unit,))
                stringRepr = self.cursor.fetchone()[0]
                print(stringRepr)
                if stringRepr.lower() == unit.lower():
                    unitMatch = 1
            except TypeError:
                continue

        return propertyMatch, unitMatch

    def computeCost(self, arr1, arr2, dist='abs'):
        # avg_arr1 = sum(arr1) / len(arr1)
        # avg_arr2 = sum(arr2) / len(arr2)
        # # variance_arr1 = sum([((x - avg_arr1) ** 2) for x in arr1]) / len(arr1)
        # # variance_arr2 = sum([((x - avg_arr2) ** 2) for x in arr2]) / len(arr1)
        # # std_v1 = variance_arr1 ** 0.5
        # # std_v2 = variance_arr2 ** 0.5
        # arr1_v2 = [x * avg_arr2 / avg_arr1 for x in arr1]
        # # arr2_v2 = [(x - avg_arr2) / std_v2 for x in arr2]
        # arr1 = arr1_v2
        # # arr2 = arr2_v2
        # arr1 always have smaller size
        if len(arr1) > len(arr2): raise ValueError('size of query column is greater than data column')

        # the algorithm may not work with float weights
        # if values are float, keep 4 decimal places
        try:
            arr1 = list(map(int, arr1))
            arr2 = list(map(int, arr2))
            factor = 1
        except ValueError:
            factor = 10000
            arr1 = [int(factor * x) for x in arr1]
            arr2 = [int(factor * x) for x in arr2]

        # time limit is 10 min
        signal.alarm(300)

        G = nx.DiGraph()
        G.add_node('s', demand=-1 * len(arr1))
        G.add_node('t', demand=len(arr1))
        for i in range(len(arr1)):
            # connect source node to all query nodes
            G.add_edge('s', i, capacity=1, weight=0)
            for j in range(len(arr2)):
                nodenum = j + len(arr1)
                # may have different distance functions
                distance = abs(arr2[j] - arr1[i])
                distance = int(10000 * distance / (abs(arr2[j]) + abs(arr1[i]))) if abs(arr2[j]) + abs(arr1[i]) != 0 else 0
                # distance = distance * int(max(arr2[j], arr1[i]) / max(1, min(arr2[j], arr1[i])))
                G.add_edge(i, nodenum, capacity=1, weight=distance)

        # connect all data nodes to sink. cost of edges are 0
        for j in range(len(arr2)):
            nodenum = j + len(arr1)
            G.add_edge(nodenum, 't', capacity=1, weight=0)

        # compute flow with min cost
        # mincostFlow = nx.max_flow_min_cost(G, 's', 't')
        mincostFlow = nx.min_cost_flow(G)
        mincost = nx.cost_of_flow(G, mincostFlow)

        # reset timer
        signal.alarm(0)

        return mincost / factor
        # return mincost/factor, mincostFlow

    def matchDistribution(self, column, distributions, method='mapping'):
        # top k nearest match
        k = 3
        costs = {}

        if method == 'mapping':
            # method using mapping function
            # allcolumns = distributions.items()
            for key in distributions:
                dataColumn = distributions[key]

                try:
                    cost = self.computeCost(column, dataColumn)
                    costs[key] = cost
                except TimeoutError:
                    continue
                except ValueError:
                    continue

            topMatch = sorted(costs.items(), key=lambda x: x[1])
            labels = [x[0] for x in topMatch[:k]]

        elif method == 'meandist':
            m = np.mean(column)
            r = np.std(column)

            for key in distributions:
                values = distributions[key]
                m2 = np.mean(values)
                diff = np.abs(m2 - m)

                if diff < r:
                    costs[key] = diff
            topMatch = sorted(costs.items(), key=lambda x: x[1])
            labels = [x[0] for x in topMatch[:k]]

        '''
        # method using statistical test
        labels = []
        alpha = 0.05
        for key in distributions:
            values = distributions[key]
            (statistic, pvalue) = mannwhitneyu(column, values, alternative="two-sided")

            if pvalue > alpha:
                labels.append(key)
        '''

        return labels

    def patternMatching(self, distribution):
        # check if column matches any pre-defined pattern. number, year, etc

        # detect rank
        distribution = sorted(distribution)
        rank = 1
        if distribution[0] == 0 or distribution[0] == 1:
            for i in range(1, len(distribution)):
                if int(distribution[i] - distribution[i - 1]) != 1:
                    rank = 0
                    break
        else:
            rank = 0

        if rank: return (True, 'rank/number',)

        # detect year
        year = 1
        for num in distribution:
            if num < 1700 or num > 2050:
                year = 0
                break

        if year: return (True, 'year',)

        # other patterns
        return (False, None)

    def getDistribution(self, eType):
        query = '''select amount from distribution where type = ? and property = ? and unit = ?; '''
        getRelationQuery = '''select type, property, unit from relation where type = ?; '''

        rv = {}
        for t in eType:
            if type(t) == str:
                self.cursor.execute(getRelationQuery, (t,))
                relations = self.cursor.fetchall()

                for relation in relations:
                    self.cursor.execute(query, relation)
                    d = []
                    row = self.cursor.fetchone()
                    while row != None:
                        d.append(row[0])
                        row = self.cursor.fetchone()

                    rv[relation] = d

            else:
                self.cursor.execute(query, t)
                d = []
                row = self.cursor.fetchone()
                while row != None:
                    if len(row[0]) != 0:
                        d.append(row[0])
                    row = self.cursor.fetchone()
                rv[t] = d

        return rv

    def getEntityType(self, entities):
        '''
        return value:
            a set containing matched entity types
            The types are represented by wikiID.
        '''

        getWikiId = '''select wid from mapping where label = ?; '''
        getType = '''select types from type where wid = ?; '''

        types = {}
        final = set()

        # thres = math.floor(len(entities) *0.5)
        # thres = math.floor(len(entities) * 0.6)
        thres = math.floor(len(entities) * 0.75)
        # thres = math.floor(len(entities) * 0.9)

        allTypes = []

        for entity in entities:
            self.cursor.execute(getWikiId, (entity,))
            wids = [x[0] for x in self.cursor.fetchall()]

            # no matching wikidata IDs with the given entity
            if wids == None or len(wids) == 0: continue

            for wid in wids:
                self.cursor.execute(getType, (wid,))
                entityType = self.cursor.fetchone()
                if entityType == None: continue

                entityType = entityType[0].split(',')

                for t in entityType:
                    if t in types:
                        types[t] += 1
                    else:
                        types[t] = 1

        if len(types) == 0:
            return None

        l = sorted(types.items(), key=lambda x: x[1], reverse=True)
        maximum = l[0][1]

        # not enough entities agree on the same type
        if maximum < thres: return None

        for t in l:
            # if t[1] == maximum:
            if t[1] >= thres:
                final.add(t[0])

        return final


class Table():
    def __init__(self):
        self.col = []
        self.header = []
        self.searchQuery = 'https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search=%s&language=en'

        self.conn = None
        self.cursor = None
        self.dbname = 'relation.db'
        self.wikitable = 'wikiTables.txt'

    def processTable(self, dataset="wikiTable"):
        p = re.compile('[a]')

        noheader = 0
        total = 0

        if dataset == 'wikiTable':
            j = 0
            with open(self.wikitable, 'w') as f:
                # for k in range(400000):
                #    line = input().strip()

                line = input().strip()

                while True:
                    # for k in range(400000):
                    # extract entity column and numeric column
                    try:
                        line = input().strip()
                        tb = json.loads(line)
                        total += 1
                    except EOFError:
                        break
                    except:
                        continue

                    index_of_columns = tb['numericColumns']

                    # no numeric columns
                    if len(index_of_columns) == 0:
                        continue

                    headers = tb['tableHeaders'][0]
                    body = tb['tableData']

                    # extract text from all cells
                    columns = []
                    numColumns = len(headers)
                    for i in range(numColumns):
                        columns.append([])

                    for rowid, rows in enumerate(body):
                        for cellNum, cell in enumerate(rows):
                            # if cell['isNumeric']:
                            index = cellNum % numColumns
                            if len(cell['text']) > 0:
                                columns[index].append(cell['text'])

                    # unique columns should have ratio = 1
                    entity = None
                    numeric = []
                    header = []
                    for i, col in enumerate(columns):
                        if len(col) == 0: continue

                        uniqueness = len(set(col)) / len(col)
                        # the left most, non numeric and most unique column
                        if (uniqueness > 0.9) and (i not in index_of_columns):
                            entity = col
                            header.append(headers[i]['text'])
                            break
                    if entity == None or entity == []: continue

                    for i in index_of_columns:
                        # print (i, len(columns), len(headers))
                        if len(columns[i]) == 0: continue
                        numeric.append(columns[i])
                        header.append(headers[i]['text'])

                    if len(header) == 1 or "" in header:
                        noheader += 1

                    # we want each table to have binary relation: one entity column and one numeric column
                    for i, col in enumerate(numeric):
                        h = [header[0], header[i + 1]]
                        table = {}
                        table['entity'] = entity
                        table['values'] = col
                        table['header'] = h

                        l = json.dumps(table)
                        f.write(l + '\n')
            print(noheader, total)

        elif dataset == "wdc":
            path = "../wdc/00/0/"

            alltbs = os.listdir(path)
            with open('wdc.txt', 'w') as outf:
                for fname in alltbs:
                    tb = None
                    table = {}
                    try:
                        with open(path + fname, 'r') as f:
                            tb = json.load(f)
                    except:
                        continue

                    # skip tables without header or key column
                    if not tb['hasHeader'] or not tb['hasKeyColumn']: continue

                    columns = tb['relation']

                    header = []
                    numeric = []

                    # extract header
                    headeridx = tb['headerRowIndex']
                    allheader = [col[headeridx] for col in columns]

                    # remove columns with empty header (for evaluation purpose only, can still annotate them)
                    if '' in allheader: continue

                    # remove header from columns
                    for i in range(len(columns)):
                        columns[i].pop(headeridx)

                    # extract entity column (key column)
                    keyidx = tb['keyColumnIndex']
                    try:
                        keyColumn = columns[keyidx]
                    except IndexError:
                        continue
                    keyheader = allheader[keyidx]

                    columns.pop(keyidx)
                    allheader.pop(keyidx)

                    # extract numeric columns
                    for i, col in enumerate(columns):
                        try:
                            c = list(map(float, col))
                            numeric.append(c)
                            header.append(allheader[i])
                        except:
                            continue

                    # no numeric columns
                    if len(numeric) == 0: continue

                    for i, col in enumerate(numeric):
                        h = [keyheader, header[i]]
                        table = {}
                        table['entity'] = keyColumn
                        table['values'] = col
                        table['header'] = h

                        l = json.dumps(table)
                        outf.write(l + '\n')

    def verify(self):
        with open(self.wikitable, 'r') as f:
            for i in range(100):
                line = f.readline().strip()
                tb = json.loads(line)
                print(json.dumps(tb, indent=4))
                print('\n\n')

    def jsonToXml(self, inputFile):
        '''
        transform data into acceptable format for KDD 14' method
        '''

        i = 0
        output = '/home/think2/Documents/dataset.xml'
        fout = open(output, 'w')
        fout.write('''<?xml version="1.0" encoding="UTF-8"?>\n<root>''')
        with open(inputFile, 'r') as f:
            line = f.readline().strip()
            while i < 180:  # line != '':
                print(line, i)
                js = json.loads(line)
                firstContent = js['values'][0]
                header = js['header'][1]
                u = js['unit']

                h = ''
                for ch in header:
                    if ch == '(' or ch == ')':
                        continue
                    elif ch.isalpha() or ch.isdigit() or ch == ' ':
                        h += ch
                    else:
                        h += ' ' + ch + ' '

                ht = str(h.split())

                xmlString = "<r>" + \
                            "<c>" + str(firstContent) + "</c>\n" + \
                            "<h>" + h + "</h>\n" + \
                            "<ht>" + ht + "</ht>\n" + \
                            "<u>" + u + "</u>\n" + \
                            "</r>"

                fout.write(xmlString + '\n')
                line = f.readline().strip()
                i += 1

        fout.write('</root>')
        fout.close()


def organizeColumns():
    with open('allColumns.pickle', 'rb') as f:
        allColumns = pickle.load(f)

    new = {}
    for key in allColumns:
        eType = key[0]
        prop = key[1]
        unit = key[2]

        v = allColumns[key]
        if eType in new:
            if prop not in new[eType]:
                new[eType][prop] = v

        else:
            new[eType] = {}
            new[eType][prop] = v

    with open('reformed.pkl', 'wb') as f:
        pickle.dump(new, f)


if __name__ == '__main__':
    # k = 5 55 27 133
    # k = 1 28 14 134
    # k = 1 11 x 67 td
    # t = Table()
    # t.processTable()
    # t.processTable(dataset="wdc")
    # t.verify()
    print("medium dataset")
    a = Annotation()
    start = time.time()
    query_column_size = None
    if len(sys.argv) == 2:
        try:
            query_column_size = int(sys.argv[1])
        except:
            query_column_size = None

    # a.annotate_separated_units()
    # print(factorial(3))
    # print("number_of_mixed_units_columns", a.number_of_mixed_units_columns)
    # print("number_of_not_mixed_units_columns", a.number_of_not_mixed_units_columns)
    a.annotate()
    end = time.time()
    print(NUM_CLUSTERS_K_MEANS)
    print(end - start)
    # count1 = 0
    # mamad = 0
    # SABET = 2
    # for i in NUM_CLUSTERS_K_MEANS:
    #     mamad += 1
    #     if mamad >= 101:
    #         SABET = 3
    #     if i == SABET:
    #         count1 += 1
    # count2 = 0
    # mamad = 0
    # SABET = 2
    # for i in NUM_CLUSTERS_NORMAL_K_MEANS:
    #     mamad += 1
    #     if mamad >= 101:
    #         SABET = 3
    #     if i == SABET:
    #         count2 += 1
    #
    # print(count1)
    # print(count2)
    # a.annotate_dbpedia()
    # a.testefficiency()
    # a.testReduction()
    # a.testReduction2()
    # a.testReduction3()
    # t.jsonToXml("dataset/unitlabels.txt")

    '''
    arr1 = [234,456, 23,135, 756, 324,21,3]
    arr2 = [np.random.randint(1000) for i in range(100)]

    cost, flow = a.computeCost(arr1, arr2)
    mset = set()
    for qnode in flow:
        if qnode == 's' or qnode == 't':
            continue

        for dnode in flow[qnode]:
            if flow[qnode][dnode] == 1:
                dnodeIdx = dnode - len(arr1)
                mset.add(arr2[dnodeIdx])

    for value in mset:
        arr2.remove(value)
    '''
