import os

def convertAnnotation(lineStr, labelMap):
    line = lineStr.strip().split(',')
    filename = line[0]
    width = int(line[1])
    height = int(line[2])

    label = line[3]
    xmin =  int(line[4])
    ymin =  int(line[5])
    xmax =  int(line[6])
    ymax =  int(line[7])

    cx = (xmin + xmax) / 2.
    cy = (ymin + ymax) / 2.

    w = xmax - xmin
    h = ymax - ymin

    if label in labelMap:
        labelCode = labelMap[label]
    else:
        labelCode = len(labelMap)
        labelMap[label] = labelCode

    return filename, (str(labelCode), str(cx/width), str(cy/height), str(w/width), str(h/height) )

def genAnnotations(filePath, outputDir, fileListName, labelMap = {}):
    annotationMap = {}

    file = open(filePath, 'r')
    lines = file.readlines()
    file.close()

    lines.remove(lines[0])

    for line in lines:
        filename, ann = convertAnnotation(line,labelMap)
        if filename not in annotationMap:
            annotationMap[filename] = []

        annotationMap[filename].append(ann)
    
    fileList = open(fileListName,'w')
    for annotationFile, annotations in annotationMap.items():
        fileList.write(annotationFile+"\n")
        annotationFile = os.path.splitext(annotationFile)[0]
        file = open(os.path.join(outputDir, annotationFile+".txt"), 'w')
        for ann in annotations:
            file.write(" ".join(ann)+"\n")
        file.close()
    fileList.close()

    return labelMap


if __name__=="__main__":
    labelMap = genAnnotations("./data/tf_record_files/train_labels.csv", "./data/dataset/train", "train.txt")
    genAnnotations("./data/tf_record_files/test_labels.csv", "./data/dataset/test", "test.txt", labelMap)

    file = open("class.name", 'w')
    sortedLabels = sorted(labelMap.items(), key=lambda x:x[1])
    for label in sortedLabels:
        file.write(label[0]+"\n")
    file.close()
