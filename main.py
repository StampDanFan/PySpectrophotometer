import pygame
import numpy as np
import sys
import json

pygame.init()

class GraphData:
    FIRST = 0.15
    SECOND = 0.65
    def __init__(self, arr, firstX=None, secondX=None):
        def get_point(x):
            if not (self.bounds[0] <= x and x < self.bounds[1]):
                return -1
            index = (x-self.bounds[0])/(self.bounds[1]-self.bounds[0])*self._origData.size
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
            self.bounds = [self.FIRST-self.firstX/full, self.SECOND+(1-self.secondX)/full]
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
                pygame.draw.line(window, "purple", (rect.left+self.FIRST*rect.width, rect.top), (rect.left+self.FIRST*rect.width, rect.bottom), 5)
                pygame.draw.line(window, "yellow", (rect.left+self.SECOND*rect.width, rect.top), (rect.left+self.SECOND*rect.width, rect.bottom), 5)

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
        image = pygame.transform.rotozoom(pygame.image.load(filepath), rotation, 1)
        self.imageArr = pygame.surfarray.array3d(image)
        self.imageArr = self.imageArr.astype(np.float64)
        self.frameRect = pygame.Rect(*topLeft, bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1])
        self.cropImage = pygame.transform.scale(pygame.surfarray.make_surface(self.imageArr[self.frameRect.left:self.frameRect.right, self.frameRect.top:self.frameRect.bottom]), (1280, 720))
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
        if self.data.bounds[0] is None:
            img = pygame.transform.scale(self.cropImage, rect.size)
            window.blit(img, rect)
        else:
            left = self.data.bounds[0] * rect.width
            right = self.data.bounds[1] * rect.width
            img = pygame.transform.scale(self.cropImage, ((right-left), rect.height))
            window.blit(img, (rect.left+left, rect.top))
        self.data.draw(window, rect, drawAlignment)

    @staticmethod
    def draw_line_graph(window, dataArr, rect):
        yArr = rect.top + (1 - dataArr) * rect.height
        xArr = rect.left + np.arange(yArr.size) / yArr.size * rect.width
        posArr = np.dstack((xArr, dataArr))[0]
        posArr = posArr[~(posArr == -1).any(axis=1)]
        posArr[:,1] = rect.top + (1 - posArr[:,1]) * rect.height
        pygame.draw.lines(window, (0, 0, 0), False, posArr)

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
        self.font = pygame.font.SysFont("toonaround", 20)
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
        pygame.event.pump()
        if out is None:
            print("User quit")
            return
        image, rotation = out
        r = self.crop_image(image)
        pygame.event.pump()
        if r is None:
            print("User quit")
            return
        graph = Graph(filepath, *r, rotation=rotation)
        out = self.standardize_graph(graph)
        pygame.event.pump()
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
        zoom = 1
        change = True
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return pygame.transform.rotozoom(originalImage, rotation, 1), rotation
                    if event.key == pygame.K_ESCAPE:
                        rotation = 0
                        change = True
                    if event.key == pygame.K_LEFT:
                        cameraPos[0] -= 20/zoom
                        change = True
                    if event.key == pygame.K_RIGHT:
                        cameraPos[0] += 20/zoom
                        change = True
                    if event.key == pygame.K_UP:
                        cameraPos[1] -= 20/zoom
                        change = True
                    if event.key == pygame.K_DOWN:
                        cameraPos[1] += 20/zoom
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
                    if event.key == pygame.K_COMMA:
                        rotation += 0.2
                        change = True
                    if event.key == pygame.K_PERIOD:
                        rotation -= 0.2
                        change = True
            if change:
                self.window.fill((0, 0, 0))
                displayImage = pygame.transform.rotozoom(originalImage, rotation, zoom)
                self.window.blit(displayImage, (-int(cameraPos[0]), -int(cameraPos[1])))
                pygame.draw.line(self.window, (0, 0, 0), (0, 360), (1280, 360), 5)

                pygame.display.flip()
                change = False

    def crop_image(self, image):
        zoom = 1
        cameraPos = [0, 0]
        leftX = None
        rightX = None
        yValue = None
        change = True

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if leftX is None:
                        leftX = (pygame.mouse.get_pos()[0]+cameraPos[0])/zoom
                        change = True
                    elif rightX is None:
                        rightX = (pygame.mouse.get_pos()[0]+cameraPos[0])/zoom
                        change = True
                    elif yValue is None:
                        yValue = (pygame.mouse.get_pos()[1]+cameraPos[1])/zoom
                        change = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if leftX is not None and rightX is not None and yValue is not None:
                            return ((leftX, yValue-2), (rightX, yValue+2))
                    if event.key == pygame.K_ESCAPE:
                        leftX = None
                        rightX = None
                        yValue = None
                        change = True
                    if event.key == pygame.K_LEFT:
                        cameraPos[0] -= 20/zoom
                        change = True
                    if event.key == pygame.K_RIGHT:
                        cameraPos[0] += 20/zoom
                        change = True
                    if event.key == pygame.K_UP:
                        cameraPos[1] -= 20/zoom
                        change = True
                    if event.key == pygame.K_DOWN:
                        cameraPos[1] += 20/zoom
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
                if leftX is not None:
                    pygame.draw.line(self.window, (255, 255, 255), (leftX * zoom - cameraPos[0], 0), (leftX * zoom - cameraPos[0], 720))
                if rightX is not None:
                    pygame.draw.line(self.window, (255, 255, 255), (rightX * zoom - cameraPos[0], 0), (rightX * zoom - cameraPos[0], 720))
                if yValue is not None:
                    pygame.draw.line(self.window, (255, 255, 255), (0, yValue * zoom - cameraPos[1]), (1280, yValue * zoom - cameraPos[1]))

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
                            if firstX and secondX:
                                graph.data.firstX = firstX / 1280
                                graph.data.secondX = secondX / 1280
                                return graph, True

                self.window.fill((0, 0, 0))
                graph.draw(self.window, pygame.Rect(0, 0, 1280, 720))
                if firstX is not None:
                    pygame.draw.line(self.window, "purple", (firstX, 0), (firstX, 720), 5)
                if secondX is not None:
                    pygame.draw.line(self.window, "yellow", (secondX, 0), (secondX, 720), 5)
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
                    pygame.draw.line(self.window, "purple", (firstX, 360), (firstX, 720), 5)
                if secondX is not None:
                    pygame.draw.line(self.window, "yellow", (secondX, 360), (secondX, 720), 5)
                pygame.display.flip()

    def show_off(self, graphs, time, overlap=False):
        self.window.fill((0, 0, 0))
        for i, graph in enumerate(graphs):
            if overlap:
                graph.draw(self.window, pygame.Rect(0, 0, 1280, 720), True)
            else:
                graph.draw(self.window, pygame.Rect(0, i*720/len(graphs), 1280, 720/len(graphs)), True)
        pygame.display.flip()
        pygame.event.pump()
        pygame.time.delay(time*1000)

    def load_graphs(self):
        with open("alignment.json") as file:
            obj = json.load(file)
        if "alignment" in obj:
            del obj["alignment"]
        return list(Graph.load(i) for i in obj), list(i for i in obj)

    def main(self):
        graphs, names = self.load_graphs()
        currGraph = 0

        change = True
        isAbsorbance = False
        absorbanceCache = None

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
                if event.type == pygame.MOUSEMOTION:
                    change = True
                if event.type == pygame.DROPFILE:
                    self.load_image(event.file, True)
                    graphs, names = self.load_graphs()
                    pygame.event.pump()
                    change = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        isAbsorbance = not isAbsorbance
                        change = True
                        if isAbsorbance:
                            absorbanceCache = graphs[currGraph].data.to_absorbance()
                        else:
                            absorbanceCache = None
                    if event.key == pygame.K_RETURN:
                        print("Reading file", names[currGraph])
                        self.load_image(names[currGraph], True)
                        pygame.event.pump()
                        change = True
                    if event.key == pygame.K_RIGHT:
                        currGraph += 1
                        change = True
                        currGraph %= len(graphs)
                        if isAbsorbance:
                            absorbanceCache = graphs[currGraph].data.to_absorbance()
                    if event.key == pygame.K_LEFT:
                        currGraph -= 1
                        change = True
                        currGraph %= len(graphs)
                        if isAbsorbance:
                            absorbanceCache = graphs[currGraph].data.to_absorbance()
            if pygame.key.get_pressed()[pygame.K_a]:
                self.window.fill((0, 0, 0))
                with open("alignment.json") as file:
                    obj = json.load(file)
                Graph.load(obj["alignment"]).draw(self.window, pygame.Rect(0, 0, 1280, 720), True)
                pygame.display.flip()
                change = True
            elif change:
                self.window.fill((200, 200, 200))
                if len(graphs) > currGraph:
                    mousePos = pygame.mouse.get_pos()
                    if isAbsorbance:
                        dataPoint = absorbanceCache.get_point(mousePos[0]/1280)
                        absorbanceCache.draw(self.window, pygame.Rect(0, 0, 1280, 720), True)
                        yText = self.font.render(f"Abs: {dataPoint:.5f}", True, (0, 0, 0))
                    else:
                        dataPoint = graphs[currGraph].data.get_point(mousePos[0] / 1280)
                        graphs[currGraph].draw(self.window, pygame.Rect(0, 0, 1280, 720), True)
                        yText = self.font.render(f"Int: {dataPoint:.5f}", True, (0, 0, 0))

                    xText = self.font.render(f"Wvl: {mousePos[0]/1280:.3f}", True, (0, 0, 0))
                    rectWidth = max(xText.get_width(), yText.get_width())+10
                    rectHeight = xText.get_height()+yText.get_height()+15
                    pygame.draw.rect(self.window, (255, 255, 255), (*mousePos, rectWidth, rectHeight))
                    self.window.blit(xText, (mousePos[0]+5, mousePos[1]+5))
                    self.window.blit(yText, (mousePos[0]+5, mousePos[1]+xText.get_height()+10))
                    pygame.draw.circle(self.window, (0, 0, 0), (mousePos[0],(1-dataPoint)*720), 3)
                    text = self.font.render(("Absorbance" if isAbsorbance else "Intensity") + " graph for " + names[currGraph], True, (255, 255, 255))
                    self.window.blit(text, (10, 10))
                change = False

                pygame.display.flip()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    g = Game()
    g.main()
