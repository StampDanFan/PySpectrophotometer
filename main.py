import pygame
import numpy as np
import sys
import json

pygame.init()

class GraphData:
    FIRST = 0.3
    SECOND = 0.5
    def __init__(self, arr, firstX=None, secondX=None):
        def get_point(x):
            if not (self.bounds[0] <= x and x < self.bounds[1]):
                return -1
            index = (self.bounds[0] + x * (self.bounds[1] - self.bounds[0])) * self._origData.size
            if index < 0 or index + 1 >= self._origData.size:
                return -1
            l = self._origData[int(index)]
            r = self._origData[int(index) + 1]
            t = index - int(index)
            return l + (r - l) * t
        self.get_point = np.vectorize(get_point, otypes=(np.float64,))

        self._origData = arr
        self.firstX = firstX
        self.secondX = secondX
        if self.firstX is not None and self.secondX is not None:
            full = (self.secondX-self.firstX)/(self.SECOND-self.FIRST)
            self.bounds = [self.firstX-full*self.FIRST, self.secondX+(1-self.SECOND)*full]
        else:
            self.bounds = [None, None]
        self.dataArr = self.transform()

    def transform(self):
        if self.firstX is not None and self.secondX is not None:
            newLength = (np.arange(1280)/1280).astype(np.float64)
            return self.get_point(newLength)
        else:
            return self._origData

    def draw(self, window, rect, drawAlignment=False):
        Graph.draw_line_graph(window, self.dataArr, rect)
        if drawAlignment == True:
            if self.firstX is not None and self.secondX is not None:
                pygame.draw.line(window, (255, 0, 0), (rect.left+self.FIRST*rect.width, rect.top), (rect.left+self.FIRST*rect.width, rect.bottom), 5)
                pygame.draw.line(window, (0, 0, 255), (rect.left+self.SECOND*rect.width, rect.top), (rect.left+self.SECOND*rect.width, rect.bottom), 5)

    def take_average(self, graphDatas):
        combArr = np.dstack((self.dataArr, *[i.dataArr for i in graphDatas], np.arange(1280)/1280))[0]
        combArr = combArr[~(combArr == -1).any(axis=1)]#combArr[np.sum(combArr != -1, axis=1)>=3]
        left, right = combArr[0][2], combArr[-1][2]
        full = right - left
        firstX = (GraphData.FIRST - left) / full
        secondX = (GraphData.SECOND - left) / full
        combArr = combArr[:, :-1]
        combArr = np.sum(combArr, axis=1)/len(graphDatas)
        return GraphData(combArr, firstX, secondX)

    def to_absorbance(self, alignmentFilepath=None):
        with open("alignment.json") as file:
            obj = json.load(file)
        if alignmentFilepath is not None:
            g = Graph.load(alignmentFilepath)
        else:
            g = Graph.load(obj["alignment"])
        combArr = np.dstack((self.dataArr, g.data.dataArr, np.arange(1280)/1280))[0]
        combArr = combArr[~(combArr == -1).any(axis=1)]
        left, right = combArr[0][2], combArr[-1][2]
        full = right-left
        firstX = (GraphData.FIRST-left)/full
        secondX = (GraphData.SECOND-left)/full
        combArr = combArr[:, :-1]
        combArr = np.divide.reduce(combArr, axis=1) # Transmittance
        combArr = -np.log(combArr) # Absorbance
        return GraphData(combArr, firstX, secondX)

class Graph:
    def __init__(self, filepath, topLeft, bottomRight, *, rotation=0, firstX=None, secondX=None):
        self.filepath = filepath
        self.rotation = rotation
        image = pygame.transform.rotate(pygame.image.load(filepath), rotation)
        self.imageArr = pygame.surfarray.array3d(image)
        self.imageArr = self.imageArr.astype(np.float64)
        self.frameRect = pygame.Rect(*topLeft, bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1])
        self.frameRect.normalize()
        self.imageArr[:,:,0] *= 0.298
        self.imageArr[:,:,1] *= 0.587
        self.imageArr[:,:,2] *= 0.114
        self.imageArr = np.add.reduce(self.imageArr, 2)
        self.imageArr /= 255

        self.data = GraphData(self.get_intensity(), firstX, secondX)

    def adjust(self, newRect):
        self.frameRect = newRect
        self.data = GraphData(self.get_intensity())

    def get_intensity(self):
        arr = self.imageArr[self.frameRect.left:self.frameRect.right, self.frameRect.top:self.frameRect.bottom]
        arr = arr.sum(axis=1)/arr.shape[1]
        return arr

    def draw(self, window, rect, drawAlignment=False):
        self.data.draw(window, rect, drawAlignment)

    @staticmethod
    def draw_line_graph(window, dataArr, rect):
        yArr = rect.top + (1 - dataArr) * rect.height
        xArr = rect.left + np.arange(yArr.size) / yArr.size * rect.width
        posArr = np.dstack((xArr, dataArr))[0]
        posArr = posArr[~(posArr == -1).any(axis=1)]
        posArr[:,1] = rect.top + (1 - posArr[:,1]) * rect.height
        pygame.draw.lines(window, "white", False, posArr)

    @staticmethod
    def load(filepath):
        with open("alignment.json") as file:
            f = json.load(file)
            if filepath in f:
                obj = f[filepath]
            else:
                print("File", filepath, "not found.")
                return None

        return Graph(obj["filepath"], obj["topleft"], obj["bottomright"], rotation=obj["rotation"], firstX=obj["firstX"], secondX=obj["secondX"])

    def save(self, filename):
        with open("alignment.json") as file:
            obj = json.load(file)
        obj[filename] = {"filepath":self.filepath,
                   "topleft":self.frameRect.topleft,
                   "bottomright":self.frameRect.bottomright,
                   "firstX":self.data.firstX,
                   "secondX":self.data.secondX,
                    "rotation":self.rotation}
        with open("alignment.json", "w") as file:
            json.dump(obj, file, indent=4)

class Game:
    def __init__(self):
        self.WIDTH = 1280
        self.HEIGHT = 720
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.image = pygame.image.load("/Users/claytonyu/Downloads/IMG_6381.jpg")
        self.frameStatus = 0
        self.frameImage = None
        pygame.display.set_caption("Spectrophotometer")
        pygame.key.set_repeat(400, 50)

    def set_calibration(self, filepath):
        with open("alignment.json", "r") as file:
            obj = json.load(file)

        if filepath not in obj:
            self.load_image(filepath)
            return

        obj["alignment"] = filepath
        with open("alignment.json", "w") as file:
            json.dump(obj, file, indent=4)
        print("Saved", filepath, "as alignment")

    def load_image(self, filepath, force=False):
        if (g := Graph.load(filepath)) and not force:
            return g
        out = self.rotate_image_file(filepath)
        if out is None:
            print("User quit")
            return
        image, rotation = out
        r = self.crop_image(image)
        if r is None:
            print("User quit")
            return
        graph = Graph(filepath, *r)
        out = self.standardize_graph(graph)
        if out is None:
            print("User quit")
            return
        graph, newAlignment = out
        if newAlignment:
            with open("alignment.json", "r") as file:
                obj = json.load(file)
            obj["alignment"] = filepath
            with open("alignment.json", "w") as file:
                json.dump(obj, file, indent=4)
            print("Saved", filepath, "as alignment")
        graph.save(filepath)
        print("Successfully saved", filepath)
        return graph

    def rotate_image_file(self, filepath):
        originalImage = pygame.image.load(filepath)
        displayImage = pygame.image.load(filepath)
        cameraPos = [0, 0]
        rotation = 0
        change = True
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return displayImage, rotation
                    if event.key == pygame.K_ESCAPE:
                        rotation = 0
                        change = True
                    if event.key == pygame.K_LEFT:
                        cameraPos[0] -= 10
                        change = True
                    if event.key == pygame.K_RIGHT:
                        cameraPos[0] += 10
                        change = True
                    if event.key == pygame.K_UP:
                        cameraPos[1] -= 10
                        change = True
                    if event.key == pygame.K_DOWN:
                        cameraPos[1] += 10
                        change = True
                    if event.key == pygame.K_MINUS:
                        rotation += 0.2
                        change = True
                    if event.key == pygame.K_EQUALS:
                        rotation -= 0.2
                        change = True
            if change:
                self.window.fill((0, 0, 0))
                displayImage = pygame.transform.rotate(originalImage, rotation)
                self.window.blit(displayImage, (-int(cameraPos[0]), -int(cameraPos[1])))
                pygame.draw.line(self.window, (0, 0, 0), (0, 360), (1280, 360), 5)

                pygame.display.flip()
                change = False

    def crop_image(self, image):
        zoom = 1
        cameraPos = [0, 0]
        firstClick = None
        secondClick = None
        change = True
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if firstClick is None:
                        firstClick = pygame.mouse.get_pos()
                        firstClick = (firstClick[0]+cameraPos[0])/zoom, (firstClick[1]+cameraPos[1])/zoom
                        change = True
                    elif secondClick is None:
                        secondClick = pygame.mouse.get_pos()
                        secondClick = (secondClick[0] + cameraPos[0]) / zoom, (secondClick[1] + cameraPos[1]) / zoom
                        change = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return (firstClick, secondClick)
                    if event.key == pygame.K_ESCAPE:
                        firstClick = None
                        secondClick = None
                        change = True
                    if event.key == pygame.K_LEFT:
                        cameraPos[0] -= 10/zoom
                        change = True
                    if event.key == pygame.K_RIGHT:
                        cameraPos[0] += 10/zoom
                        change = True
                    if event.key == pygame.K_UP:
                        cameraPos[1] -= 10/zoom
                        change = True
                    if event.key == pygame.K_DOWN:
                        cameraPos[1] += 10/zoom
                        change = True
                    if event.key == pygame.K_MINUS:
                        zoom -= 0.05
                        change = True
                        if zoom < 0.1:
                            zoom = 0.1
                    if event.key == pygame.K_EQUALS:
                        zoom += 0.05
                        change = True
                        if zoom > 10:
                            zoom = 10
            if change:
                self.window.fill((0, 0, 0))
                displayImage = pygame.transform.scale(image, (int(image.get_width()*zoom), int(image.get_height()*zoom)))
                self.window.blit(displayImage, (-int(cameraPos[0]), -int(cameraPos[1])))
                if firstClick is not None:
                    if secondClick is not None:
                        topleft = (firstClick[0] * zoom - cameraPos[0], firstClick[1] * zoom - cameraPos[1])
                        bottomright = (secondClick[0] * zoom - cameraPos[0], secondClick[1] * zoom - cameraPos[1])
                        pygame.draw.rect(self.window, (255, 255, 255), (*topleft, bottomright[0]-topleft[0], bottomright[1]-topleft[1]), 4)
                    else:
                        pygame.draw.circle(self.window, (255, 255, 255), (firstClick[0]*zoom-cameraPos[0], firstClick[1]*zoom-cameraPos[1]), 2)

                pygame.display.flip()
                change = False

    def standardize_graph(self, graph):
        alignGraph = None

        with open("alignment.json") as file:
            obj = json.load(file)
            if "alignment" in obj:
                alignGraph = Graph.load(obj["alignment"])
        if alignGraph is None:
            print("No alignment graph")
            firstX = None
            secondX = None
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if firstX is None:
                            firstX = pygame.mouse.get_pos()[0]
                        elif secondX is None:
                            secondX = pygame.mouse.get_pos()[0]
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            firstX = None
                            secondX = None
                        if event.key == pygame.K_RETURN:
                            graph.data.firstX = firstX / 1280
                            graph.data.secondX = secondX / 1280
                            return graph, True

                self.window.fill((0, 0, 0))
                graph.draw(self.window, pygame.Rect(0, 0, 1280, 720))
                if firstX is not None:
                    pygame.draw.line(self.window, (255, 0, 0), (firstX, 0), (firstX, 720), 5)
                if secondX is not None:
                    pygame.draw.line(self.window, (0, 0, 255), (secondX, 0), (secondX, 720), 5)
                pygame.display.flip()
        else:
            print("Alignment graph found")
            firstX = None
            secondX = None
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if firstX is None:
                            firstX = pygame.mouse.get_pos()[0]
                        elif secondX is None:
                            secondX = pygame.mouse.get_pos()[0]
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            firstX = None
                            secondX = None
                        if event.key == pygame.K_RETURN:
                            graph.data.firstX = firstX / 1280
                            graph.data.secondX = secondX / 1280
                            return graph, False

                self.window.fill((0, 0, 0))
                alignGraph.draw(self.window, pygame.Rect(0, 0, 1280, 360), True)
                graph.draw(self.window, pygame.Rect(0, 360, 1280, 360))
                if firstX is not None:
                    pygame.draw.line(self.window, (255, 0, 0), (firstX, 360), (firstX, 720), 5)
                if secondX is not None:
                    pygame.draw.line(self.window, (0, 0, 255), (secondX, 360), (secondX, 720), 5)
                pygame.display.flip()

    def show_off(self, graphs, time, overlap=False):
        self.window.fill((0, 0, 0))
        for i, graph in enumerate(graphs):
            if overlap:
                graph.draw(self.window, pygame.Rect(0, 0, 1280, 720), True)
            else:
                graph.draw(self.window, pygame.Rect(0, i*720/len(graphs), 1280, 720/len(graphs)), True)
        pygame.display.flip()
        pygame.event.get()
        pygame.time.delay(time*1000)

    def load_graphs(self):
        with open("alignment.json") as file:
            obj = json.load(file)
        del obj["alignment"]
        return list(Graph.load(i) for i in obj), list(i for i in obj)

    def main(self):
        self.set_calibration("Sucrose/a- 100 Water/a1 Large.jpeg")
        graphs, names = self.load_graphs()
        currGraph = 0

        change = True
        isAbsorbance = False

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
                if event.type == pygame.DROPFILE:
                    self.load_image(event.file, True)
                    graphs, names = self.load_graphs()
                    pygame.event.pump()
                    change = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        isAbsorbance = not isAbsorbance
                        change = True
                    if event.key == pygame.K_RETURN:
                        print(names[currGraph], "with absorbance:", isAbsorbance)
                    if event.key == pygame.K_RIGHT:
                        currGraph += 1
                        change = True
                        currGraph %= len(graphs)
                    if event.key == pygame.K_LEFT:
                        currGraph -= 1
                        change = True
                        currGraph %= len(graphs)
            if pygame.key.get_pressed()[pygame.K_a]:
                self.window.fill((0, 0, 0))
                with open("alignment.json") as file:
                    obj = json.load(file)
                Graph.load(obj["alignment"]).draw(self.window, pygame.Rect(0, 0, 1280, 720), True)
                pygame.display.flip()
                change = True
            elif change:
                self.window.fill((0, 0, 0))
                if isAbsorbance:
                    g = graphs[currGraph].data.to_absorbance()
                    np.set_printoptions(threshold=sys.maxsize)
                    g.draw(self.window, pygame.Rect(0, 0, 1280, 720), True)
                else:
                    graphs[currGraph].draw(self.window, pygame.Rect(0, 0, 1280, 720), True)
                change = False

                pygame.display.flip()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    g = Game()
    g.main()
