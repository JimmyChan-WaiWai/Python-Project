
import itchat
#登录微信
itchat.auto_login(enableCmdQR=-1)#enableCmdQR在终端或命令行中为True,在notebook中为-1

def sendMessageToWechat(markName=u'喵喵',message=u'已经处理完毕'):
    '''
    markName: 微信备注的名字
    message: 要发送的内容
    eg: sendMessageToWechat(markName=u'鹏举',message=u'已经处理完毕')
    '''
    #想给谁发信息，先查找到这个朋友
    users = itchat.search_friends(name=markName)
    if users:
        #找到UserName
        userName = users[0]['UserName']
        itchat.send(message,toUserName = userName)
    else:
        print('通讯录中无此人')


from time import sleep

def func1():
    sleep(20)
def func2():
    sleep(40)

func1()
sendMessageToWechat(markName=u'喵喵',message=u'func1已经处理完毕')
func2()
sendMessageToWechat(markName=u'喵喵',message=u'func2已经处理完毕')



from selenium import webdriver
import time
#使用chrome的webdriver
browser = webdriver.Chrome()
#開啟google首頁

browser.get('http://sscsummerevent2020.com/game_start.php#main')
x=input("Press 1 if ready: ")
if x=="1":
    print("OK")
    for i in range(1,100):
        content = browser.find_elements_by_class_name('des_bold')[3].text
        if(content=="很可惜"):
            browser.find_elements_by_class_name("btn_lucky_draw")[0].click()
            time.sleep(2.8)

